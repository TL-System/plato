"""Replay local container-helper degradation checks without Docker execution."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

root = Path(sys.argv[1]).resolve()
artifacts = Path(sys.argv[2]).resolve()
artifacts.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "container_checks", root / ".github/scripts/check_container.py"
)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
records = []


def reject(name, operation, expected):
    try:
        operation()
    except RuntimeError as error:
        assert expected in str(error), error
        records.append({"case": name, "accepted": True,
                        "expected_rejection": str(error), "simulated": True})
    else:
        raise AssertionError(f"{name}: invalid input was accepted")


read_text = Path.read_text
ignore_file = root / ".dockerignore"


def mutated_ignore(extra=None, remove=None):
    def read(path, *args, **kwargs):
        text = read_text(path, *args, **kwargs)
        if path == ignore_file:
            if remove:
                text = "\n".join(line for line in text.splitlines() if line != remove)
            if extra:
                text += f"\n{extra}\n"
        return text
    return read


for name, extra, remove, error in (
    ("global-toml-exclusion", "*.toml", None, "Build source excluded: pyproject.toml"),
    ("workspace-source-exclusion", "examples/detector", None,
     "Build source excluded: examples/detector/pyproject.toml"),
    ("evidence-context-leak", None, "evidence", "Not excluded: evidence/container-check"),
):
    with patch.object(Path, "read_text", mutated_ignore(extra, remove)):
        reject(name, lambda: checks.check_static(root), error)

assert checks.read_probe("CUDA banner\n" + checks.PROBE_PREFIX + '{"accepted":true}\n')["accepted"]
records.append({"case":"normal-NVIDIA-banner", "accepted":True, "simulated":True})
reject("missing-probe-record", lambda: checks.read_probe("CUDA banner\n"),
       "Expected one marked container probe record")
reject("duplicate-probe-record", lambda: checks.read_probe(
    checks.PROBE_PREFIX + '{}\n' + checks.PROBE_PREFIX + '{}\n'),
    "Expected one marked container probe record")

disk = namedtuple("disk", "total used free")
fake_docker = {"DockerRootDir":"/simulated-docker-root", "OSType":"linux",
               "Architecture":"x86_64"}
checker = checks.LinuxCheck(root, artifacts / "simulated-resource-gate")
checker.artifacts.mkdir()


def fake_run(argv, name, **kwargs):
    assert argv[0] in {"docker", "df"}, argv
    return json.dumps(fake_docker) if argv[:2] == ["docker", "info"] else ""


with patch.object(checker, "run", fake_run):
    with patch.object(shutil, "disk_usage", return_value=disk(100, 86, 14 * 1024**3)):
        reject("standard-14GiB-resource-floor", lambda: checker.resources("before"),
               "Need at least 35 GiB free")
    with patch.object(shutil, "disk_usage", return_value=disk(100, 55, 45 * 1024**3)):
        measured = checker.resources("before")
        records.append({"case":"45GiB-resource-floor", "accepted":True,
                        "measured":measured, "simulated":True})
        fake_docker["Architecture"] = "aarch64"
        reject("wrong-daemon-architecture", lambda: checker.resources("before"),
               "Qualification requires a Linux amd64 Docker daemon")

with patch.dict(os.environ, {"RUNNER_ENVIRONMENT":"self-hosted"}):
    with patch.object(checker, "run", side_effect=AssertionError("No cleanup allowed")):
        checker.reclaim_hosted_sdks({"disks":{"fixture":{"free":14 * 1024**3}}})
cleanup = json.loads((checker.artifacts / "hosted-sdk-reclamation.json").read_text())
assert cleanup["removed"] == [] and "outside a GitHub-hosted" in cleanup["reason"]
records.append({"case":"no-SDK-removal-on-self-hosted-or-user-host", "accepted":True,
                "record":cleanup, "simulated":True})

# Run the real CLI guard on the macOS user host. It must fail before Docker.
assert sys.platform == "darwin"
assert shutil.which("docker") is None
negative = artifacts / "local-linux-refusal"
process = subprocess.run([
    sys.executable, str(root / ".github/scripts/check_container.py"), "linux",
    "--root", str(root), "--artifacts", str(negative),
], capture_output=True, text=True)
(artifacts / "local-linux-refusal.log").write_text(process.stdout + process.stderr)
receipt = json.loads((negative / "linux-acceptance.json").read_text())
assert process.returncode == 1 and receipt["accepted"] is False
assert "only on Linux amd64 CI" in receipt["error"]
assert not (negative / "commands.json").exists()
records.append({"case":"real-macOS-linux-mode-refusal", "accepted":True,
                "simulated":False, "exit_code":process.returncode,
                "docker_commands_executed":False, "receipt":receipt})

(artifacts / "negative-checks.json").write_text(json.dumps({
    "accepted":True, "source":checks.source_record(root), "cases":records,
    "scope":"Local static/helper degradation only; no actual image or GPU proof",
}, indent=2)+"\n")
print(f"Passed {len(records)} helper/degradation checks without Docker execution")
