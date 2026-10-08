
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path
import yaml

repository = Path("/tmp/plato-refresh-worktrees/docs-package-refresh").resolve()
negative_repository = Path("/tmp/plato-refresh-worktrees/docs-package-negative").resolve()
artifact_dir = Path("/tmp/plato-p2-final")
negative_dir = artifact_dir / "negative"
negative_dir.mkdir(parents=True, exist_ok=True)
source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
python = sys.executable
spec = importlib.util.spec_from_file_location("checker", repository / ".github/scripts/check_distribution.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
results = []

def expected_failure(label, action, substring):
    try:
        action()
    except Exception as error:
        assert substring in str(error), (label, error)
        results.append({"name": label, "accepted": True, "expected_rejection": str(error)})
    else:
        raise AssertionError("Negative input was accepted: " + label)

def execute(label, command, cwd=repository, environment=None, expected=None):
    if command[0] not in ("zsh", python):
        raise AssertionError(command)
    process = subprocess.run(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, check=False)
    (negative_dir / (label + ".log")).write_text(process.stdout)
    assert process.returncode != 0, (label, process.stdout)
    if expected:
        assert expected in process.stdout, (label, process.stdout)
    results.append({"name": label, "accepted": True, "exit_code": process.returncode,
                    "expected_rejection": expected, "command": command, "cwd": str(cwd)})

project = tomllib.loads((repository / "pyproject.toml").read_text())["project"]
sources = set(subprocess.check_output(["git", "ls-files", "plato/**/*.py", "plato/*.py"],
                                     cwd=repository, text=True).splitlines())
original_red = Path("/tmp/plato-p2-preliminary/package-second/distributions/plato_learn-1.4.3.tar.gz")
expected_failure("actual_pre_fix_archived_sdist",
                 lambda: checker.inspect_distribution(original_red, project, sources),
                 "Forbidden distribution payloads")
results[-1]["artifact_sha256"] = checker.sha256(original_red)
results[-1]["origin"] = json.loads(Path("/tmp/plato-p2-preliminary/package-second/provenance.json").read_text())
# Use the actual positive wheel when available; do not wait by polling.
wheel = Path("/tmp/plato-p2-preliminary/package-third/distributions/plato_learn-1.4.3-py3-none-any.whl")
for label, added, removed, replacement, expected in (
    ("archived_wheel", {"archives/legacy/module.py": b"sentinel"}, set(), {}, "Forbidden"),
    ("reproduction_wheel", {"evidence/2026-refresh/reproductions/sentinel.txt": b"sentinel"}, set(), {}, "Forbidden"),
    ("environment_wheel", {".venv/lib/sentinel": b"sentinel"}, set(), {}, "Forbidden"),
    ("cache_wheel", {"__pycache__/sentinel.pyc": b"sentinel"}, set(), {}, "Forbidden"),
    ("missing_source_wheel", {}, {"plato/config.py"}, {}, "Missing Plato source"),
    ("wrong_version_metadata", {}, set(), {"Version: 1.4.3": "Version: 0.0.0"}, "Version:"),
    ("wrong_python_metadata", {}, set(), {"Requires-Python: >=3.13": "Requires-Python: >=3.14"}, "Requires-Python:"),
):
    directory = negative_dir / label
    directory.mkdir(exist_ok=True)
    mutated = directory / wheel.name
    with zipfile.ZipFile(wheel) as incoming, zipfile.ZipFile(mutated, "w") as outgoing:
        for name in incoming.namelist():
            if name in removed:
                continue
            payload = incoming.read(name)
            if name.endswith(".dist-info/METADATA"):
                for before, after in replacement.items():
                    assert before.encode() in payload
                    payload = payload.replace(before.encode(), after.encode())
            outgoing.writestr(name, payload)
        for name, payload in added.items():
            outgoing.writestr(name, payload)
    expected_failure(label, lambda: checker.inspect_distribution(mutated, project, sources), expected)
    results[-1]["artifact_sha256"] = checker.sha256(mutated)

dirty = negative_dir / "existing_output"
dirty.mkdir(exist_ok=True)
(dirty / "acceptance.json").write_text("preserve-this-existing-receipt\n")
execute("existing_output_rejected", [python, str(repository / ".github/scripts/check_distribution.py"),
                                    "--output-dir", str(dirty)], expected="File exists")
assert (dirty / "acceptance.json").read_text() == "preserve-this-existing-receipt\n"

fake_bin = negative_dir / "wrong_uv"
fake_bin.mkdir(exist_ok=True)
(fake_bin / "uv").write_text("#!/bin/sh\nprintf 'uv 0.12.21\\n'\n")
(fake_bin / "uv").chmod(0o755)
environment = os.environ.copy()
environment["PATH"] = str(fake_bin) + ":" + environment["PATH"]
wrong_output = negative_dir / "wrong_uv_output"
execute("wrong_uv_rejected", [python, str(repository / ".github/scripts/check_distribution.py"),
                             "--output-dir", str(wrong_output)], environment=environment,
        expected="uv 0.12.22 is required")
assert not wrong_output.exists()

# A source checkout as cwd must fail before importing any package dependencies.
manifest = artifact_dir / "package" / "wheel-manifest.json"
execute("repository_cwd_rejected",
        [python, "-I", "-c", checker.IMPORT_PROBE, str(repository), str(manifest),
         project["version"], str(negative_dir / "unexpected-probe.json")],
        expected="AssertionError")
assert not (negative_dir / "unexpected-probe.json").exists()

environment = os.environ.copy()
environment["PLATO_DOCS_PYTHON"] = python
environment["PLATO_DOCS_ENVIRONMENT"] = str(negative_repository / ".venv")
execute("application_environment_rejected", ["zsh", "-lc", "./docs/build.sh"],
        cwd=negative_repository, environment=environment, expected="AssertionError")
assert not (negative_repository / ".venv").exists()

requirements = negative_repository / "docs/requirements.txt"
requirements.write_text(requirements.read_text() + "# deliberate parity mismatch\n")
environment["PLATO_DOCS_ENVIRONMENT"] = str(negative_dir / "parity_environment")
execute("requirements_parity_rejected", ["zsh", "-lc", "./docs/build.sh"],
        cwd=negative_repository, environment=environment, expected="EOF")
assert not (negative_dir / "parity_environment").exists()
subprocess.run(["git", "restore", "docs/requirements.txt"], cwd=negative_repository, check=True)

# Parse action inputs and every shell/Python command without executing publication.
for filename in (".github/workflows/docs_package_checks.yml", ".github/workflows/pypi_publish.yml"):
    document = yaml.load((repository / filename).read_text(), Loader=yaml.BaseLoader)
    assert document["permissions"] == {"contents": "read"}
    if filename.endswith("pypi_publish.yml"):
        assert document["on"] == {"release": {"types": ["created"]}}
    else:
        assert "release" not in document["on"]
        assert all(not re.search(r"\buv\s+publish\b", step.get("run", "")) for job in document["jobs"].values()
                   for step in job["steps"])
    for job in document["jobs"].values():
        assert job["defaults"]["run"]["shell"] == 'zsh -lc ". {0}"'
        for step in job["steps"]:
            if "uses" in step:
                assert len(step["uses"].split("@")[1]) == 40
            if "run" in step:
                if step.get("shell") == "python {0}":
                    ast.parse(step["run"])
                else:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".zsh") as handle:
                        handle.write(step["run"])
                        handle.flush()
                        subprocess.run(["zsh", "-n", handle.name], check=True)
    results.append({"name": filename, "accepted": True, "yaml_and_embedded_commands_parsed": True})

configuration = tomllib.loads((repository / "netlify.toml").read_text())
assert configuration["build"] == {"base": ".", "command": "./docs/build.sh", "publish": "docs/site",
                                  "environment": {"PYTHON_VERSION": "3.13"}}
subprocess.run(["sh", "-n", str(repository / "docs/build.sh")], check=True)
subprocess.run([python, "-m", "py_compile", str(repository / ".github/scripts/check_distribution.py")], check=True)
results.append({"name": "netlify_toml_and_source_syntax", "accepted": True})
receipt = {"schema_version": 1, "source_commit": source_commit, "accepted": True,
           "checker_sha256": checker.sha256(repository / ".github/scripts/check_distribution.py"),
           "results": results}
(negative_dir / "checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"accepted": True, "checks": len(results), "source_commit": source_commit}))
