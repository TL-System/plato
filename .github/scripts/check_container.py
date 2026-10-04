"""Check launchers locally; qualify the actual CPU image only on Linux CI."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

BASE_IMAGE = (
    "nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:"
    "b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266"
)
UV_IMAGE = (
    "ghcr.io/astral-sh/uv:0.12.22@sha256:"
    "f513a91fc62fe7c17567eee97230dd198e43edb8a9fbecca843714a4358fe1bc"
)
SOURCE_FILES = (
    "Dockerfile",
    ".dockerignore",
    "dockerrun.sh",
    "dockerrun_gpu.sh",
    ".github/workflows/container_checks.yml",
    ".github/scripts/check_container.py",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
)
SENTINEL = ".plato-container-exclusion-sentinel"
EXCLUDED_PATHS = (
    "archives/retired",
    "evidence/container-check",
    ".venv",
    ".venv-host",
    ".cache",
    "plato/__pycache__",
    "source-cache",
    "source-caches",
    "examples/detector/.venv",
    "examples/detector/.venv-host",
)
MIN_FREE_BYTES = 35 * 1024**3
PROBE_PREFIX = "PLATO_CONTAINER_PROBE_JSON="
HOSTED_SDK_PATHS = (
    ("android", Path("/usr/local/lib/android")),
    ("dotnet", Path("/usr/share/dotnet")),
    ("swift", Path("/usr/share/swift")),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, record: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_probe(output: str) -> dict[str, object]:
    # NVIDIA's normal entrypoint prints a CUDA banner before executing commands.
    records = [
        line[len(PROBE_PREFIX) :]
        for line in output.splitlines()
        if line.startswith(PROBE_PREFIX)
    ]
    require(len(records) == 1, "Expected one marked container probe record.")
    return json.loads(records[0])


def ignored(path: str, patterns: list[str]) -> bool:
    """Evaluate this file's positive-only patterns for static context screening.

    The actual Docker build and in-image sentinel checks remain authoritative.
    """
    ancestors = [str(p) for p in Path(path).parents if str(p) != "."]
    for pattern in patterns:
        alternatives = [pattern]
        if pattern.startswith("**/"):
            alternatives.append(pattern[3:])
        if any(
            fnmatch.fnmatchcase(candidate, alternative)
            for candidate in [path, *ancestors]
            for alternative in alternatives
        ):
            return True
    return False


def source_record(root: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {
        "commit": commit,
        "file_sha256": {name: sha256(root / name) for name in SOURCE_FILES},
        "base_image": BASE_IMAGE,
        "uv_image": UV_IMAGE,
        "host_python": sys.version,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }


def check_static(root: Path) -> dict[str, object]:
    dockerfile = (root / "Dockerfile").read_text()
    require(BASE_IMAGE in dockerfile, "Selected CUDA image digest must be pinned.")
    require(UV_IMAGE in dockerfile, "Selected uv image digest must be pinned.")
    patterns = [
        line.strip().rstrip("/")
        for line in (root / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    require(
        not any(p.startswith("!") for p in patterns),
        "Static screening needs extension before adding ignore negations.",
    )
    project = tomllib.loads((root / "pyproject.toml").read_text())
    members = project["tool"]["uv"]["workspace"]["members"]
    manifests = [f"{member}/pyproject.toml" for member in members]
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    required = [
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "README.md",
        "LICENSE",
        "plato.py",
        *manifests,
    ] + [
        name for name in tracked if name.startswith(("plato/", "configs/", "examples/"))
    ]
    for name in required:
        require((root / name).is_file(), f"Missing required build source: {name}")
        require(not ignored(name, patterns), f"Build source excluded: {name}")
    for name in (*EXCLUDED_PATHS, ".git", "tests", "runtime", "models", "data"):
        require(ignored(f"{name}/{SENTINEL}", patterns), f"Not excluded: {name}")

    # A recording stub checks actual shell argv and failure propagation. It
    # neither contacts a Docker daemon nor makes a GPU qualification claim.
    cases = []
    with tempfile.TemporaryDirectory(prefix="plato launcher ") as temporary:
        directory = Path(temporary).resolve()
        stub = directory / "docker"
        stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['PLATO_DOCKER_RECORD']).write_text("
            "json.dumps(sys.argv[1:]))\n"
            "raise SystemExit(int(os.environ['PLATO_DOCKER_EXIT']))\n"
        )
        stub.chmod(0o755)
        record = directory / "argv.json"
        environment = dict(os.environ, PATH=f"{directory}:{os.environ['PATH']}")
        environment["PLATO_DOCKER_RECORD"] = str(record)
        command = ["python", "-c", "literal $x `x` $(x)", "a b", ""]
        for script, gpu in (("dockerrun.sh", False), ("dockerrun_gpu.sh", True)):
            require(
                (root / script).stat().st_mode & 0o111 != 0,
                f"Launcher must be executable: {script}",
            )
            subprocess.run(["sh", "-n", str(root / script)], check=True)
            for arguments, exit_code in (([], 0), (command, 0), (command, 23)):
                environment["PLATO_DOCKER_EXIT"] = str(exit_code)
                result = subprocess.run(
                    [str(root / script), *arguments],
                    cwd=directory,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                )
                expected = [
                    "run",
                    "--rm",
                    "--net=host",
                    "-v",
                    "/dev/shm:/dev/shm",
                    "-v",
                    f"{directory}:/root/plato",
                    *(["--gpus", "all"] if gpu else []),
                    "-i",
                    "plato",
                    *(arguments or ["/bin/bash"]),
                ]
                actual = json.loads(record.read_text())
                require(actual == expected, f"Incorrect {script} argv: {actual}")
                require(result.returncode == exit_code, "Launcher lost exit status.")
                cases.append({"script": script, "argv": actual, "exit": exit_code})
    return {
        "accepted": True,
        "scope": "Static context and recording-stub checks only",
        "workspace_manifests": manifests,
        "required_source_count": len(required),
        "launcher_cases": cases,
        "gpu_compute_tested": False,
    }


def inside_probe(expected_lock: str, mounted: bool) -> dict[str, object]:
    """Run offline CPU, installed-environment and library provenance assertions."""
    root = Path("/root/plato")
    environment = Path("/opt/plato/.venv")
    require(Path(sys.prefix).resolve() == environment, "Wrong Python environment.")
    require(Path(sys.executable).parent == environment / "bin", "Wrong executable.")
    require(sys.version_info[:2] == (3, 13), "Python 3.13 is required.")
    require(
        sys.dont_write_bytecode,
        "Container imports must not write bytecode into the mounted checkout.",
    )
    require(
        Path(sys.base_prefix).is_relative_to("/opt/plato/python"),
        "Managed interpreter must be outside the checkout mount.",
    )
    require(
        os.environ.get("UV_PROJECT_ENVIRONMENT") == str(environment),
        "Incorrect uv environment selection.",
    )
    require(sha256(root / "uv.lock") == expected_lock, "Checkout lock mismatch.")
    require(
        Path("/opt/plato/lock.sha256").read_text().split()[0] == expected_lock,
        "Built image lock mismatch.",
    )
    project = tomllib.loads((root / "pyproject.toml").read_text())
    manifests = [
        f"{member}/pyproject.toml"
        for member in project["tool"]["uv"]["workspace"]["members"]
    ]
    for name in ["LICENSE", "README.md", "plato.py", *manifests]:
        require((root / name).is_file(), f"Missing image source: {name}")
    if not mounted:
        for name in EXCLUDED_PATHS:
            require(not (root / name / SENTINEL).exists(), f"Leaked context: {name}")
        require(not (root / "archives").exists(), "Archives leaked into the image.")
        require(not (root / "evidence").exists(), "Evidence leaked into the image.")
        require(not (root / ".git").exists(), "Git checkout leaked into the image.")
    modules = [
        "torch",
        "torchvision",
        "plato",
        "numpy",
        "scipy",
        "aiohttp",
        "accelerate",
        "datasets",
        "evaluate",
        "peft",
        "transformers",
        "tenseal",
        "socketio",
        "torch_optimizer",
        "timm",
        "zstd",
    ]
    imports = {name: importlib.import_module(name).__file__ for name in modules}
    for name in ("torch", "torchvision", "numpy"):
        require(
            Path(imports[name]).is_relative_to(environment),
            f"Import outside installed environment: {name}",
        )
    require(Path(imports["plato"]).is_relative_to(root), "Wrong editable source.")
    lock = tomllib.loads((root / "uv.lock").read_text())
    versions = {package["name"]: package["version"] for package in lock["package"]}
    distributions = {}
    for name in (
        "plato-learn",
        "torch",
        "torchvision",
        "cuda-toolkit",
        "nvidia-cudnn-cu13",
    ):
        distribution = importlib.metadata.distribution(name)
        require(distribution.version == versions[name], f"Lock version drift: {name}")
        distributions[name] = {
            "version": distribution.version,
            "location": str(distribution.locate_file("")),
            "direct_url": distribution.read_text("direct_url.json"),
        }
    origin = json.loads(distributions["plato-learn"]["direct_url"])
    require(origin["url"] == "file:///root/plato", "Wrong project install origin.")
    require(origin["dir_info"]["editable"], "Development install must be editable.")
    import torch
    import torchvision

    require(
        torch.version.cuda == ".".join(versions["cuda-toolkit"].split(".")[:2]),
        "Unexpected torch CUDA build.",
    )
    tensor = torch.tensor([1.0, 2.0, 3.0])
    require(
        (tensor.square() == torch.tensor([1.0, 4.0, 9.0])).all().item(),
        "CPU tensor operation failed.",
    )
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist()
    require(kept == [0], "Torchvision native CPU NMS operation failed.")
    major, minor, patch, *_ = map(int, versions["nvidia-cudnn-cu13"].split("."))
    cudnn = torch.backends.cudnn.version()
    require(
        cudnn == major * 10000 + minor * 100 + patch,
        "Loaded cuDNN does not match its installed wheel version.",
    )
    libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line and re.search(r"cuda|cudnn|cublas|cufft|nvrtc|nccl", line)
        }
    )
    cudnn_distribution = importlib.metadata.distribution("nvidia-cudnn-cu13")
    cudnn_files = {
        str(cudnn_distribution.locate_file(file).resolve())
        for file in cudnn_distribution.files or []
        if "libcudnn" in str(file)
    }
    loaded_cudnn = [name for name in libraries if "libcudnn" in name]
    require(bool(loaded_cudnn), "cuDNN version probe did not load a library.")
    require(
        all(str(Path(name).resolve()) in cudnn_files for name in loaded_cudnn),
        "Loaded cuDNN must come from the locked Python wheel.",
    )
    uv_version = subprocess.check_output(["uv", "--version"], text=True).strip()
    require(uv_version.startswith("uv 0.12.22 "), "Unexpected uv version.")
    pip_check = subprocess.run(
        ["uv", "pip", "check", "--python", sys.executable],
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        "accepted": True,
        "scope": "Linux amd64 CPU imports and startup",
        "mounted": mounted,
        "hostname": socket.gethostname(),
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "dont_write_bytecode": sys.dont_write_bytecode,
        "uv": uv_version,
        "lock_sha256": expected_lock,
        "imports": imports,
        "workspace_manifests": manifests,
        "distributions": distributions,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": cudnn,
        "loaded_cuda_cudnn_libraries": libraries,
        "torchvision_cpu_nms": kept,
        "uv_pip_check": pip_check.stderr,
        "package_freeze": subprocess.check_output(
            ["uv", "pip", "freeze", "--python", sys.executable], text=True
        ),
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "os_release": Path("/etc/os-release").read_text(),
        "os_packages": subprocess.check_output(
            ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"], text=True
        ),
        "gpu_compute_tested": False,
    }


class LinuxCheck:
    def __init__(self, root: Path, artifacts: Path) -> None:
        self.root = root
        self.artifacts = artifacts
        self.commands: list[dict[str, object]] = []

    def run(
        self,
        argv: list[str],
        name: str,
        *,
        cwd: Path | None = None,
        expected: int = 0,
        timeout: int = 180,
    ) -> str:
        started = time.monotonic()
        output = self.artifacts / f"{name}.log"
        record: dict[str, object] = {"argv": argv, "cwd": str(cwd or self.root)}
        self.commands.append(record)
        try:
            with output.open("w") as log:
                result = subprocess.run(
                    argv,
                    cwd=cwd or self.root,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            record["exit_code"] = result.returncode
            require(
                result.returncode == expected,
                f"{name} exited {result.returncode}; see {output}",
            )
            return output.read_text()
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["log"] = output.name
            save(self.artifacts / "commands.json", self.commands)

    def resources(self, phase: str) -> dict[str, object]:
        info = json.loads(
            self.run(
                ["docker", "info", "--format", "{{json .}}"], f"docker-info-{phase}"
            )
        )
        paths = [self.root, Path(info["DockerRootDir"])]
        disks = {str(path): shutil.disk_usage(path)._asdict() for path in paths}
        self.run(["df", "-h", *map(str, paths)], f"df-{phase}")
        self.run(["docker", "system", "df"], f"docker-disk-{phase}")
        record = {
            "phase": phase,
            "disks": disks,
            "minimum_free_bytes": MIN_FREE_BYTES,
            "base_compressed_bytes": 3973716026,
            "standard_runner_documented_storage_gb": 14,
            "runner_os": platform.platform(),
            "docker_os": info["OSType"],
            "docker_arch": info["Architecture"],
        }
        save(self.artifacts / f"resources-{phase}.json", record)
        require(
            info["OSType"] == "linux" and info["Architecture"] == "x86_64",
            "Qualification requires a Linux amd64 Docker daemon.",
        )
        if phase == "before":
            require(
                all(disk["free"] >= MIN_FREE_BYTES for disk in disks.values()),
                "Need at least 35 GiB free for CUDA/PyTorch build and layers. "
                "Select an available larger Linux x64 PLATO_CONTAINER_RUNNER; "
                "do not substitute a CPU image or claim a passed build.",
            )
        return record

    def require_removed(self, hostname: str, name: str) -> None:
        output = self.run(
            ["docker", "container", "inspect", hostname],
            f"{name}-auto-removal",
            expected=1,
        )
        require("No such" in output, "Container removal was not proved.")

    def reclaim_hosted_sdks(self, initial: dict[str, object]) -> None:
        """Reclaim only unused documented SDKs on a disposable hosted runner."""
        record: dict[str, object] = {"removed": [], "skipped": []}
        if all(disk["free"] >= MIN_FREE_BYTES for disk in initial["disks"].values()):
            record["reason"] = "Initial measured disk already meets the build floor."
        elif os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted":
            record["reason"] = "Never remove SDKs outside a GitHub-hosted runner."
        elif platform.freedesktop_os_release().get("VERSION_ID") != "24.04" or (
            platform.freedesktop_os_release().get("ID") != "ubuntu"
        ):
            record["reason"] = "SDK locations are documented for Ubuntu 24.04 only."
        else:
            record["reason"] = "Initial disk below floor; reclaim fixed unused SDKs."
            # Locations are documented by actions/runner-images' Ubuntu 24.04
            # inventory and install-dotnetcore-sdk.sh / install-swift.sh.
            for name, path in HOSTED_SDK_PATHS:
                if not path.is_dir() or path.is_symlink():
                    record["skipped"].append(str(path))
                    continue
                self.run(["sudo", "du", "-sk", "--", str(path)], f"sdk-{name}-size")
                self.run(
                    ["sudo", "rm", "-rf", "--", str(path)],
                    f"sdk-{name}-remove",
                    timeout=120,
                )
                require(not path.exists(), f"SDK reclamation failed: {path}")
                record["removed"].append(str(path))
        save(self.artifacts / "hosted-sdk-reclamation.json", record)

    def qualify(self) -> None:
        require(
            platform.system() == "Linux" and platform.machine() == "x86_64",
            "Actual image checks run only on Linux amd64 CI.",
        )
        require(
            os.environ.get("GITHUB_ACTIONS") == "true",
            "Use the GitHub Linux workflow; do not run containers on the host.",
        )
        initial = self.resources("initial")
        self.reclaim_hosted_sdks(initial)
        self.resources("before")
        source = source_record(self.root)
        save(self.artifacts / "source.json", source)
        lock_hash = sha256(self.root / "uv.lock")
        for name in EXCLUDED_PATHS:
            sentinel = self.root / name / SENTINEL
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text(f"Excluded from {source['commit']}: {name}\n")
        save(self.artifacts / "context-sentinels.json", list(EXCLUDED_PATHS))
        for name, image in (("base", BASE_IMAGE), ("uv", UV_IMAGE)):
            self.run(
                ["docker", "buildx", "imagetools", "inspect", image], f"{name}-manifest"
            )
            self.run(
                ["docker", "buildx", "imagetools", "inspect", "--raw", image],
                f"{name}-manifest-raw",
            )
        self.run(
            [
                "docker",
                "build",
                "--pull",
                "--progress=plain",
                "--platform",
                "linux/amd64",
                "--iidfile",
                str(self.artifacts / "image-id.txt"),
                "--build-arg",
                f"PLATO_SOURCE_COMMIT={source['commit']}",
                "-t",
                "plato",
                ".",
            ],
            "build",
            timeout=2700,
        )
        inspect = json.loads(
            self.run(["docker", "image", "inspect", "plato"], "image-inspect")
        )[0]
        image_id = (self.artifacts / "image-id.txt").read_text().strip()
        require(inspect["Id"] == image_id, "Built and inspected image IDs differ.")
        require(
            inspect["Config"]["Labels"]["org.opencontainers.image.revision"]
            == source["commit"],
            "Image source revision is incorrect.",
        )
        save(
            self.artifacts / "image-summary.json",
            {
                "id": image_id,
                "size_bytes": inspect["Size"],
                "source": source,
            },
        )
        self.run(["docker", "history", "--no-trunc", "plato"], "image-history")
        probe = self.root / ".github/scripts/check_container.py"
        output = self.run(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "-e",
                "CUDA_VISIBLE_DEVICES=",
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "HF_DATASETS_OFFLINE=1",
                "--mount",
                f"type=bind,src={probe},dst=/tmp/check_container.py,readonly",
                "plato",
                "python",
                "/tmp/check_container.py",
                "inside",
                "--expected-lock",
                lock_hash,
            ],
            "unmounted-cpu",
            timeout=300,
        )
        cpu = read_probe(output)
        save(self.artifacts / "unmounted-cpu.json", cpu)
        self.require_removed(cpu["hostname"], "unmounted")
        help_output = self.run(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "plato",
                "python",
                "plato.py",
                "--help",
            ],
            "cli-help",
            timeout=180,
        )
        require("--config" in help_output, "Plato CLI did not produce expected help.")
        self.check_mount(lock_hash)

    def check_mount(self, lock_hash: str) -> None:
        with tempfile.TemporaryDirectory(prefix="plato mounted checkout ") as temporary:
            fixture = Path(temporary).resolve()
            checkout = fixture / "source with spaces"
            shutil.copytree(
                self.root,
                checkout,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".venv*",
                    "archives",
                    "evidence",
                    "__pycache__",
                    ".cache",
                ),
            )
            host_env = checkout / ".venv"
            (host_env / "bin").mkdir(parents=True)
            (host_env / "pyvenv.cfg").write_text("invalid host environment\n")
            (host_env / "bin/python").write_text("#!/bin/sh\nexit 97\n")
            (host_env / "bin/python").chmod(0o755)
            before = {
                str(p.relative_to(host_env)): sha256(p)
                for p in host_env.rglob("*")
                if p.is_file()
            }
            output = self.run(
                [
                    str(self.root / "dockerrun.sh"),
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    ".github/scripts/check_container.py",
                    "inside",
                    "--expected-lock",
                    lock_hash,
                    "--mounted",
                ],
                "mounted-cpu",
                cwd=checkout,
                timeout=300,
            )
            mounted = read_probe(output)
            self.require_removed(mounted["hostname"], "mounted")
            code = (
                "import json,socket,sys; "
                f"print({PROBE_PREFIX!r}+json.dumps({{'hostname':socket.gethostname(), "
                "'args':sys.argv[1:]})); raise SystemExit(23)"
            )
            arguments = ["a b", "literal $x `x` $(x)", ""]
            output = self.run(
                [
                    str(self.root / "dockerrun.sh"),
                    "python",
                    "-c",
                    code,
                    *arguments,
                ],
                "mounted-exit",
                cwd=checkout,
                expected=23,
            )
            exit_record = read_probe(output)
            require(
                exit_record["args"] == arguments, "Real argument forwarding failed."
            )
            self.require_removed(exit_record["hostname"], "mounted-exit")
            after = {
                str(p.relative_to(host_env)): sha256(p)
                for p in host_env.rglob("*")
                if p.is_file()
            }
            require(before == after, "Container modified the invalid host environment.")
            bytecode_paths = [
                str(path.relative_to(checkout))
                for path in checkout.rglob("__pycache__")
            ]
            require(
                not bytecode_paths, "Smoke imports wrote bytecode into the fixture."
            )
            record = {
                "accepted": True,
                "cpu": mounted,
                "forwarding": exit_record,
                "host_venv_before": before,
                "host_venv_after": after,
                "command_exit_code": 23,
                "auto_removed": True,
                "bytecode_paths": bytecode_paths,
            }
        # The mount receipt includes successful ordinary host fixture cleanup.
        # Cleanup errors propagate; no ownership changes or privileged deletion.
        require(not fixture.exists(), "Mounted checkout fixture was not removed.")
        record["fixture_cleanup_complete"] = True
        save(self.artifacts / "mounted-cpu.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("static", "linux", "inside"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--expected-lock")
    parser.add_argument("--mounted", action="store_true")
    args = parser.parse_args()
    if args.mode == "inside":
        require(bool(args.expected_lock), "Inside probe requires the source lock hash.")
        print(PROBE_PREFIX + json.dumps(inside_probe(args.expected_lock, args.mounted)))
        return
    require(args.artifacts is not None, "An artifact directory is required.")
    root, artifacts = args.root.resolve(), args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, object] = {
        "accepted": False,
        "mode": args.mode,
        "gpu_compute_tested": False,
    }
    checker = LinuxCheck(root, artifacts)
    started = time.monotonic()
    try:
        save(artifacts / "source.json", source_record(root))
        if args.mode == "static":
            receipt.update(check_static(root))
        else:
            checker.qualify()
            receipt.update(accepted=True, scope="Actual Linux amd64 CPU image checks")
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if args.mode == "linux" and checker.commands:
            try:
                checker.resources("after")
            except Exception as error:
                receipt["final_resource_error"] = str(error)
                receipt["accepted"] = False
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save(artifacts / f"{args.mode}-acceptance.json", receipt)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in receipt.items()
                    if key
                    in {
                        "accepted",
                        "mode",
                        "scope",
                        "error",
                        "elapsed_seconds",
                        "final_resource_error",
                    }
                }
            )
        )
    require(receipt["accepted"] is True, "Qualification failed; inspect artifacts.")


if __name__ == "__main__":
    main()
