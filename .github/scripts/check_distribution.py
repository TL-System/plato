"""Build locked distributions and prove imports from a separately installed wheel."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

UV_VERSION = "0.12.22"
SOURCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".github/scripts/check_distribution.py",
    ".github/workflows/docs_package_checks.yml",
    ".github/workflows/pypi_publish.yml",
)

IMPORT_PROBE = r"""
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

repository, wheel_manifest, expected_version, receipt_path = sys.argv[1:]
repository = Path(repository).resolve()
prefix = Path(sys.prefix).resolve()
assert sys.version_info[:2] == (3, 13), sys.version
assert not Path.cwd().resolve().is_relative_to(repository), Path.cwd()
distribution = importlib.metadata.distribution("plato-learn")
assert distribution.version == expected_version, distribution.version
installed_root = Path(distribution.locate_file(".")).resolve()
assert installed_root.is_relative_to(prefix), installed_root
assert not installed_root.is_relative_to(repository), installed_root
manifest = json.loads(Path(wheel_manifest).read_text())["files"]
for name, digest in manifest.items():
    if not name.endswith(".dist-info/RECORD"):
        installed_file = Path(distribution.locate_file(name)).resolve()
        assert installed_file.is_relative_to(prefix), installed_file
        assert hashlib.sha256(installed_file.read_bytes()).hexdigest() == digest, name
imports = {}
for name in (
    "plato", "plato.config", "plato.utils.tree", "torch", "torchvision",
    "numpy", "scipy", "aiohttp", "socketio", "datasets", "transformers",
):
    module = importlib.import_module(name)
    location = Path(module.__file__).resolve()
    assert location.is_relative_to(prefix), (name, location, prefix)
    assert not location.is_relative_to(repository), (name, location)
    imports[name] = str(location)
assert importlib.import_module("plato").__version__ == expected_version
import torch
import torchvision
assert torch.equal(torch.tensor([1, 2]) + 1, torch.tensor([2, 3]))
kept = torchvision.ops.nms(
    torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]]),
    torch.tensor([1., 0.5]), 0.5,
)
assert kept.tolist() == [0], kept
packages = {
    item.metadata["Name"]: item.version
    for item in importlib.metadata.distributions()
}
direct_url = json.loads(distribution.read_text("direct_url.json"))
assert "archive_info" in direct_url and "dir_info" not in direct_url, direct_url
receipt = {
    "python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
    "cwd": str(Path.cwd()), "distribution_root": str(installed_root),
    "version": distribution.version, "imports": imports,
    "direct_url": direct_url,
    "installed_payload_matches_wheel": True,
    "cpu_tensor_and_torchvision_operation": True,
    "packages": dict(sorted(packages.items())),
}
Path(receipt_path).write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
"""


def sha256(path: Path) -> str:
    """Return a file's SHA256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbidden_payload(name: str) -> bool:
    """Identify generated, archived, cache and environment payloads."""
    parts = PurePosixPath(name).parts
    excluded = {
        "archives",
        "reproductions",
        "__pycache__",
        ".git",
        ".cache",
        ".pytest_cache",
        ".ruff_cache",
        "source-cache",
        "source-caches",
        "ci-artifacts",
    }
    return parts[:2] == ("docs", "site") or any(
        part in excluded or part.startswith(".venv") for part in parts
    )


def check_metadata(payload: bytes, project: dict) -> dict[str, object]:
    """Require distribution identity and interpreter metadata from the project."""
    metadata = BytesParser().parsebytes(payload)
    for field, key in (
        ("Name", "name"),
        ("Version", "version"),
        ("Requires-Python", "requires-python"),
    ):
        if metadata[field] != project[key]:
            raise ValueError(f"{field}: {metadata[field]!r} != {project[key]!r}")
    return {
        "name": metadata["Name"],
        "version": metadata["Version"],
        "requires_python": metadata["Requires-Python"],
        "requires_dist": metadata.get_all("Requires-Dist", []),
        "provides_extra": metadata.get_all("Provides-Extra", []),
    }


def inspect_distribution(
    path: Path, project: dict, expected_sources: set[str]
) -> dict[str, object]:
    """Inspect the actual distribution file manifest and core source coverage."""
    files = {}
    metadata_payloads = []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith("/"):
                    continue
                files[name] = hashlib.sha256(archive.read(name)).hexdigest()
                if name.endswith(".dist-info/METADATA"):
                    metadata_payloads.append(archive.read(name))
    else:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                name = PurePosixPath(member.name)
                relative = PurePosixPath(*name.parts[1:]).as_posix()
                payload = archive.extractfile(member).read()
                files[relative] = hashlib.sha256(payload).hexdigest()
                if relative == "PKG-INFO":
                    metadata_payloads.append(payload)
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            if name not in files:
                raise ValueError(f"Missing sdist input: {name}")
    forbidden = sorted(name for name in files if forbidden_payload(name))
    if forbidden:
        raise ValueError(f"Forbidden distribution payloads: {forbidden}")
    missing = sorted(expected_sources - files.keys())
    if missing:
        raise ValueError(f"Missing Plato source files: {missing}")
    if len(metadata_payloads) != 1:
        raise ValueError(f"Expected one metadata record, got {len(metadata_payloads)}")
    normalized = project["name"].replace("-", "_")
    prefix = f"{normalized}-{project['version']}"
    valid_name = (
        path.name.startswith(prefix + "-")
        if path.suffix == ".whl"
        else path.name == prefix + ".tar.gz"
    )
    if not valid_name:
        raise ValueError(f"Unexpected distribution filename: {path.name}")
    return {
        "filename": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "metadata": check_metadata(metadata_payloads[0], project),
        "files": dict(sorted(files.items())),
    }


def run(command: list[str], cwd: Path, log: Path) -> str:
    """Retain command output and propagate failures."""
    with log.open("w") as output:
        output.write(f"cwd: {cwd}\ncommand: {json.dumps(command)}\n")
        output.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=output,
            stderr=subprocess.STDOUT,
            env={
                key: value for key, value in os.environ.items() if key != "PYTHONPATH"
            },
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
    return log.read_text()


def validate_package(repository: Path, output: Path) -> dict[str, object]:
    """Build, rebuild, inspect and install with explicit lock constraints."""
    if sys.version_info[:2] != (3, 13):
        raise ValueError(f"Python 3.13 is required: {sys.version}")
    uv_version = subprocess.check_output(["uv", "--version"], text=True).strip()
    if uv_version.split()[1] != UV_VERSION:
        raise ValueError(f"uv {UV_VERSION} is required: {uv_version}")
    if output.is_relative_to(repository) and not output.is_relative_to(
        repository / "ci-artifacts"
    ):
        raise ValueError(
            "Output within the checkout must be under the excluded ci-artifacts path."
        )
    output.mkdir(parents=True, exist_ok=False)
    project = tomllib.loads((repository / "pyproject.toml").read_text())["project"]
    sources = set(
        subprocess.check_output(
            ["git", "ls-files", "plato/**/*.py", "plato/*.py"],
            cwd=repository,
            text=True,
        ).splitlines()
    )
    provenance = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            text=True,
        ),
        "source_files": {name: sha256(repository / name) for name in SOURCE_FILES},
        "python": sys.version,
        "platform": platform.platform(),
        "uv": uv_version,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    build_constraints = output / "build-constraints.txt"
    runtime_constraints = output / "runtime-constraints.txt"
    export = [
        "uv",
        "export",
        "--locked",
        "--python",
        "3.13",
        "--no-emit-project",
        "--no-header",
        "--no-hashes",
    ]
    run(
        export + ["--only-group", "build", "--output-file", str(build_constraints)],
        repository,
        output / "build-constraints.log",
    )
    run(
        export + ["--no-default-groups", "--output-file", str(runtime_constraints)],
        repository,
        output / "runtime-constraints.log",
    )
    distributions = output / "distributions"
    build = [
        "uv",
        "build",
        "--no-sources",
        "--build-constraints",
        str(build_constraints),
        "--python",
        "3.13",
    ]
    run(build + ["--out-dir", str(distributions)], repository, output / "build.log")
    wheels = list(distributions.glob("*.whl"))
    sdists = list(distributions.glob("*.tar.gz"))
    artifacts = [path for path in distributions.iterdir() if path.name != ".gitignore"]
    if len(wheels) != 1 or len(sdists) != 1 or len(artifacts) != 2:
        raise ValueError("Expected exactly one wheel and one sdist in clean output.")
    wheel = inspect_distribution(wheels[0], project, sources)
    sdist = inspect_distribution(sdists[0], project, sources)
    wheel_manifest = output / "wheel-manifest.json"
    wheel_manifest.write_text(json.dumps(wheel, indent=2) + "\n")
    (output / "sdist-manifest.json").write_text(json.dumps(sdist, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="plato-package-") as temporary:
        temporary = Path(temporary)
        with tarfile.open(sdists[0], "r:gz") as archive:
            archive.extractall(temporary / "source", filter="data")
        roots = list((temporary / "source").iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError("Expected a single source root in the sdist.")
        rebuilt_dir = output / "rebuilt"
        run(
            build + ["--wheel", "--out-dir", str(rebuilt_dir), "."],
            roots[0],
            output / "sdist-rebuild.log",
        )
        rebuilt_wheels = list(rebuilt_dir.glob("*.whl"))
        if len(rebuilt_wheels) != 1:
            raise ValueError("Expected one wheel rebuilt from the sdist.")
        rebuilt = inspect_distribution(rebuilt_wheels[0], project, sources)
        (output / "rebuilt-wheel-manifest.json").write_text(
            json.dumps(rebuilt, indent=2) + "\n"
        )
        if (
            rebuilt["metadata"] != wheel["metadata"]
            or rebuilt["files"] != wheel["files"]
        ):
            raise ValueError("Rebuilt wheel metadata or payload differs from original.")
        environment = temporary / "environment"
        run(
            ["uv", "venv", "--python", "3.13", str(environment)],
            temporary,
            output / "wheel-environment.log",
        )
        python = environment / "bin/python"
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--constraint",
                str(runtime_constraints),
                str(wheels[0]),
            ],
            temporary,
            output / "wheel-install.log",
        )
        run(
            ["uv", "pip", "check", "--python", str(python)],
            temporary,
            output / "wheel-dependencies.log",
        )
        probe_log = output / "installed-wheel-import.log"
        run(
            [
                str(python),
                "-I",
                "-c",
                IMPORT_PROBE,
                str(repository),
                str(wheel_manifest),
                project["version"],
                str(output / "installed-wheel.json"),
            ],
            temporary,
            probe_log,
        )
        probe = json.loads((output / "installed-wheel.json").read_text())
        locked = tomllib.loads((repository / "uv.lock").read_text())
        versions = {}
        for package in locked["package"]:
            versions.setdefault(package["name"], set()).add(package["version"])
        for name, version in probe["packages"].items():
            normalized = re.sub(r"[-_.]+", "-", name).lower()
            if version not in versions.get(normalized, set()):
                raise ValueError(
                    f"Installed dependency is not locked: {name}=={version}"
                )
        (output / "installed-wheel.json").write_text(json.dumps(probe, indent=2) + "\n")
    return {
        "accepted": True,
        "source_commit": provenance["source_commit"],
        "wheel_sha256": wheel["sha256"],
        "sdist_sha256": sdist["sha256"],
        "rebuilt_wheel_payload_matches": True,
        "installed_wheel_provenance_and_imports": True,
        "build_constraints_sha256": sha256(build_constraints),
        "runtime_constraints_sha256": sha256(runtime_constraints),
    }


def main() -> int:
    """Write a success or failure receipt for CI and release validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ci-artifacts/docs-package")
    )
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    output = arguments.output_dir.resolve()
    try:
        receipt = validate_package(repository, output)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        if output.is_dir() and not (output / "acceptance.json").exists():
            (output / "acceptance.json").write_text(
                json.dumps({"accepted": False, "error": str(error)}, indent=2) + "\n"
            )
        return 1
    (output / "acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
