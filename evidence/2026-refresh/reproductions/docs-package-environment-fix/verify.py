import ast
import hashlib
import json
import subprocess
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path("/tmp/plato-p2-docs-environment-fix")
SOURCE = Path("/tmp/plato-refresh-worktrees/docs-package-refresh").resolve()
COMMIT = "06ba679690f090aacf7e6a95a6dffd74b81ffad3"
project = tomllib.loads((SOURCE / "pyproject.toml").read_text())
results = {}
def normalize(text):
    parsed = Requirement(text)
    return (canonicalize_name(parsed.name), tuple(sorted(parsed.extras)),
            str(parsed.specifier), str(parsed.marker) if parsed.marker else None)
for label, path in (
    ("default", Path("/tmp/plato-refresh-worktrees/docs-package-env-default")),
    ("bootstrap", Path("/tmp/plato-refresh-worktrees/docs-package-env-bootstrap")),
    ("release", Path("/tmp/plato-refresh-worktrees/docs-package-env-release")),
):
    package = path / "ci-artifacts/docs-package"
    accepted = json.loads((package / "acceptance.json").read_text())
    binding = json.loads((package / "sdist-source-binding.json").read_text())
    sdist = json.loads((package / "sdist-manifest.json").read_text())
    wheel = json.loads((package / "wheel-manifest.json").read_text())
    assert accepted["source_commit"] == binding["source_commit"] == COMMIT
    assert accepted["sdist_source_tree_bound"] and accepted["accepted"]
    assert accepted["sdist_source_binding_sha256"] == hashlib.sha256((package / "sdist-source-binding.json").read_bytes()).hexdigest()
    assert len(binding["files"]) == binding["git_tree_bound_members"] == len(sdist["files"]) - 1
    assert set(binding["files"]) == set(sdist["files"]) - {"PKG-INFO"}
    assert all(record["sha256"] == sdist["files"][name] for name, record in binding["files"].items())
    assert all(not name.startswith((".venv-docs/", ".venv-docs-bootstrap/", "docs/site/", "ci-artifacts/")) for name in sdist["files"])
    with zipfile.ZipFile(package / "distributions" / wheel["filename"]) as archive:
        metadata = BytesParser().parsebytes(archive.read(next(name for name in archive.namelist()
                                                              if name.endswith(".dist-info/METADATA"))))
    declarations = set(map(normalize, project["project"]["dependencies"]))
    for extra, dependencies in project["project"]["optional-dependencies"].items():
        for dependency in dependencies:
            declarations.add(normalize(dependency + '; extra == "' + canonicalize_name(extra) + '"'))
    assert set(map(normalize, metadata.get_all("Requires-Dist"))) == declarations
    assert set(metadata.get_all("Provides-Extra")) == set(map(canonicalize_name, project["project"]["optional-dependencies"]))
    assert metadata["Requires-Python"] == project["project"]["requires-python"]
    installed = json.loads((package / "installed-wheel.json").read_text())
    wheel_path = (package / "distributions" / wheel["filename"]).resolve()
    assert Path(unquote(urlsplit(installed["direct_url"]["url"]).path)).resolve() == wheel_path
    assert installed["installed_payload_matches_wheel"] and installed["cpu_tensor_and_torchvision_operation"]
    assert not Path(installed["cwd"]).resolve().is_relative_to(path.resolve())
    assert all(Path(location).resolve().is_relative_to(Path(installed["prefix"]).resolve())
               for location in installed["imports"].values())
    rebuilt = json.loads((package / "rebuilt-wheel-manifest.json").read_text())
    assert wheel["metadata"] == rebuilt["metadata"] and wheel["files"] == rebuilt["files"]
    for workspace in project["tool"]["uv"]["workspace"]["members"]:
        assert workspace + "/pyproject.toml" in sdist["files"]
    results[label] = {
        "source_commit": COMMIT, "accepted": True, "checkout": str(path),
        "command": ["uv", "run", "--no-project", "--python", "3.13", "python", ".github/scripts/check_distribution.py"],
        "sdist_sha256": accepted["sdist_sha256"], "wheel_sha256": accepted["wheel_sha256"],
        "git_tree_bound_members": binding["git_tree_bound_members"],
        "only_generated_exception": binding["generated_metadata"],
        "requires_dist_extras_and_requires_python_parity": True,
        "clean_rebuild_and_installed_wheel_proof": True,
    }
assert len({build["sdist_sha256"] for build in results.values()}) == 1
assert len({build["wheel_sha256"] for build in results.values()}) == 1
docs = json.loads(Path("/tmp/plato-refresh-worktrees/docs-package-env-default/ci-artifacts/docs/docs-environment.json").read_text())
assert docs["requirements_parity"] and docs["runtime_and_project_absent"]
assert all(name not in docs["packages"] for name in ("torch", "lighteval", "plato-learn"))
exports = ROOT / "requirements-parity.txt"
subprocess.run(["uv", "export", "--locked", "--python", "3.13", "--only-group", "docs",
                "--no-emit-project", "--no-header", "--output-file", str(exports)], cwd=SOURCE,
               stdout=subprocess.DEVNULL, check=True)
assert exports.read_bytes() == (SOURCE / "docs/requirements.txt").read_bytes()
for filename in (".github/workflows/docs_package_checks.yml", ".github/workflows/pypi_publish.yml"):
    workflow = yaml.load((SOURCE / filename).read_text(), Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {"contents": "read"}
    for job in workflow["jobs"].values():
        assert job["defaults"]["run"]["shell"] == 'zsh -lc ". {0}"'
        for step in job["steps"]:
            if "run" in step:
                if step.get("shell") == "python {0}":
                    ast.parse(step["run"])
                else:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".zsh") as script:
                        script.write(step["run"])
                        script.flush()
                        subprocess.run(["zsh", "-n", script.name], check=True)
    if filename.endswith("pypi_publish.yml"):
        assert workflow["on"] == {"release": {"types": ["created"]}}
configuration = tomllib.loads((SOURCE / "netlify.toml").read_text())
assert configuration["build"] == {"base": ".", "command": "./docs/build.sh", "publish": "docs/site",
                                  "environment": {"PYTHON_VERSION": "3.13"}}
(ROOT / "default-validation.json").write_text(json.dumps({
    "source_commit": COMMIT, "accepted": True, "builds": results,
    "fresh_docs_requirement_parity": True, "docs_dependency_count": len(docs["packages"]),
    "workflow_and_netlify_syntax_and_publication_boundary": True,
    "default_and_bootstrap_environments_retained_and_excluded": True,
}, indent=2) + "\n")
print(json.dumps({"accepted": True, "source_commit": COMMIT,
                  "sdist_sha256": results["default"]["sdist_sha256"],
                  "git_tree_bound_members": results["default"]["git_tree_bound_members"]}))

# Independently compare actual tar member bytes with a Git archive of the frozen
# tree. Keep all dirty generated directories and environment symlinks on disk.
import io
import tarfile
archive_bytes = subprocess.check_output(["git", "archive", "--format=tar", COMMIT], cwd=SOURCE)
with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as git_archive:
    frozen_bytes = {member.name: git_archive.extractfile(member).read()
                    for member in git_archive.getmembers() if member.isfile()}
independent = {}
for label, path in (("default", Path("/tmp/plato-refresh-worktrees/docs-package-env-default")),
                    ("bootstrap", Path("/tmp/plato-refresh-worktrees/docs-package-env-bootstrap")),
                    ("release", Path("/tmp/plato-refresh-worktrees/docs-package-env-release"))):
    package = path / "ci-artifacts/docs-package"
    manifest = json.loads((package / "sdist-manifest.json").read_text())
    sdist_path = package / "distributions" / manifest["filename"]
    with tarfile.open(sdist_path) as archive:
        count = 0
        for member in archive.getmembers():
            if member.isdir():
                continue
            assert member.isfile(), member.name
            relative = Path(*Path(member.name).parts[1:]).as_posix()
            payload = archive.extractfile(member).read()
            assert hashlib.sha256(payload).hexdigest() == manifest["files"][relative]
            if relative != "PKG-INFO":
                assert frozen_bytes[relative] == payload, relative
                count += 1
    independent[label] = {"git_archive_source": COMMIT, "all_regular_source_member_bytes_equal": True,
                          "members": count, "sdist_sha256": hashlib.sha256(sdist_path.read_bytes()).hexdigest()}
    assert not (path / ".venv").exists()
    if label == "release":
        assert not (path / ".venv-docs").exists() and not (path / ".venv-docs-bootstrap").exists()
        continue
    docs_env = json.loads((path / "ci-artifacts/docs/docs-environment.json").read_text())
    assert docs_env["requirements_parity"] and docs_env["runtime_and_project_absent"]
    assert Path(docs_env["prefix"]).resolve() == (path / ".venv-docs").resolve()
    assert (path / ".venv-docs/bin/python").is_symlink()
    assert (path / ".venv-docs/p2-m3-environment-sentinel.json").is_file()
    assert (path / "docs/site/p2-m3-site-sentinel.json").is_file()
    assert (path / "ci-artifacts/older/p2-m3-artifact-sentinel.json").is_file()
    after = json.loads((ROOT / (label + "-environment-after.json")).read_text())
    assert after["untouched_by_package_checker"]
    if label == "bootstrap":
        python = path / ".venv-docs-bootstrap/bin/python"
        inventory = json.loads(subprocess.check_output([str(python), "-c",
                    "import importlib.metadata,json; print(json.dumps({d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()}))"], text=True))
        assert inventory["uv"] == "0.12.22"
        assert not {"plato-learn", "torch", "lighteval"} & set(inventory)
        version = subprocess.check_output([str(path / ".venv-docs-bootstrap/bin/uv"), "--version"], text=True).strip()
        assert version.split()[1] == "0.12.22"
        (ROOT / "bootstrap-inventory.json").write_text(json.dumps({"packages": inventory,
            "uv_version": version, "default_bootstrap_prefix": str(path / ".venv-docs-bootstrap"),
            "no_project_or_runtime": True, "uv_was_absent_from_docs_path": True,
            "docs_path": "/usr/bin:/bin:/usr/sbin:/sbin"}, indent=2) + "\n")
previous = tomllib.loads(subprocess.check_output(["git", "show", "cd40f3f862782ccf22fd95d5c7ba1822328842f9:pyproject.toml"], cwd=SOURCE, text=True))
current = tomllib.loads((SOURCE / "pyproject.toml").read_text())
exclusions = current["tool"]["hatch"]["build"]["exclude"]
assert exclusions == [".venv-docs/**", ".venv-docs-bootstrap/**"] + previous["tool"]["hatch"]["build"]["exclude"]
current["tool"]["hatch"]["build"]["exclude"] = previous["tool"]["hatch"]["build"]["exclude"]
assert current == previous
for filename in ("uv.lock", ".github/scripts/check_distribution.py", "docs/build.sh",
                 "docs/requirements.txt", "netlify.toml", ".github/workflows/pypi_publish.yml",
                 ".github/workflows/docs_package_checks.yml"):
    assert (SOURCE / filename).read_bytes() == subprocess.check_output(
        ["git", "show", "cd40f3f862782ccf22fd95d5c7ba1822328842f9:" + filename], cwd=SOURCE)
(ROOT / "independent-source-bytes.json").write_text(json.dumps({
    "source_commit": COMMIT, "accepted": True, "builds": independent,
    "pyproject_only_two_authorized_exclusions": True,
    "lock_dependencies_markers_versions_and_existing_p2_files_unchanged": True,
}, indent=2) + "\n")
