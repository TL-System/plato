"""Verify M1 bytecode/cleanup behavior with real child imports and no Docker."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

root = Path(sys.argv[1]).resolve()
outputs = Path(sys.argv[2]).resolve()
outputs.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "container_checks", root / ".github/scripts/check_container.py"
)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
value = re.search(r"^\s*PYTHONDONTWRITEBYTECODE=([^\s]+)",
                  (root / "Dockerfile").read_text(), re.M)[1]
assert value == "1"
records = []

# Actual startup behavior in a new interpreter, including a red control.
with tempfile.TemporaryDirectory(prefix="plato bytecode import ") as temporary:
    directory = Path(temporary).resolve()
    module = directory / "bytecode_probe.py"
    module.write_text("value = 42\n")
    for enabled in (False, True):
        environment = dict(os.environ)
        environment.pop("PYTHONPYCACHEPREFIX", None)
        environment.pop("PYTHONDONTWRITEBYTECODE", None)
        if enabled:
            environment["PYTHONDONTWRITEBYTECODE"] = value
        child = subprocess.run([
            sys.executable, "-c", "import bytecode_probe,json,sys; "
            "print(json.dumps({'flag':sys.dont_write_bytecode, "
            "'value':bytecode_probe.value}))",
        ], cwd=directory, env=environment, capture_output=True, text=True, check=True)
        result = json.loads(child.stdout)
        bytecode = list(directory.rglob("*.pyc"))
        assert result == {"flag":enabled,"value":42}
        assert bool(bytecode) is not enabled
        records.append({"case":"actual-child-import", "bytecode_disabled":enabled,
                        "result":result,"bytecode":list(map(str,bytecode)),
                        "simulated":False})
        if bytecode:
            shutil.rmtree(directory / "__pycache__")


class FixtureCheck(checks.LinuxCheck):
    """Exercise the real mount checker; simulate only Docker/launcher transport."""

    def __init__(self, artifacts, suppress):
        super().__init__(root, artifacts)
        self.fixture = None
        self.child_results = []
        self.suppress = suppress
        artifacts.mkdir()

    def run(self, argv, name, *, cwd=None, expected=0, timeout=180):
        if "auto-removal" in name:
            assert expected == 1
            return "Error: No such container: simulated-container\n"
        self.fixture = cwd.parent
        environment = dict(os.environ)
        environment.pop("PYTHONPYCACHEPREFIX", None)
        environment.pop("PYTHONDONTWRITEBYTECODE", None)
        if self.suppress:
            environment["PYTHONDONTWRITEBYTECODE"] = value
        if name == "mounted-cpu":
            # plato.__init__ is dependency-free. Import the copied real package
            # under the actual fixture path containing spaces.
            code = "import plato,json,sys; print(" + repr(checks.PROBE_PREFIX) + \
                "+json.dumps({'accepted':True,'hostname':'simulated-container'," \
                "'dont_write_bytecode':sys.dont_write_bytecode}))"
            command = [sys.executable, "-c", code]
        elif name == "mounted-exit":
            command = [sys.executable, *argv[2:]]
        else:
            raise AssertionError(name)
        process = subprocess.run(command, cwd=cwd, env=environment,
                                 capture_output=True, text=True)
        assert process.returncode == expected, process.stderr
        self.child_results.append({"command":command,"cwd":str(cwd),
                                   "exit_code":process.returncode,
                                   "stdout":process.stdout})
        return process.stdout


for suppress in (False, True):
    checker = FixtureCheck(outputs / ("suppressed" if suppress else "control"), suppress)
    try:
        checker.check_mount("unused-transport-simulation-lock")
    except RuntimeError as error:
        assert not suppress and "Smoke imports wrote bytecode" in str(error), error
        assert not (checker.artifacts / "mounted-cpu.json").exists()
        result = {"expected_rejection":str(error)}
    else:
        assert suppress
        result = json.loads((checker.artifacts / "mounted-cpu.json").read_text())
        assert result["accepted"] and result["fixture_cleanup_complete"]
        assert result["cpu"]["dont_write_bytecode"] is True
        assert result["bytecode_paths"] == []
    assert not checker.fixture.exists()
    records.append({"case":"real-mount-checker-with-child-package-import",
                    "bytecode_disabled":suppress,"result":result,
                    "child_processes":checker.child_results,
                    "transport_simulated":True,"docker_executed":False})

# The remote symptom: cleanup PermissionError must prevent a passing receipt.
# Simulate ownership denial without root files, chmod/chown or sudo on the host.
checker = FixtureCheck(outputs / "cleanup-denial", True)
with patch.object(tempfile.TemporaryDirectory, "_rmtree",
                  side_effect=PermissionError("simulated root-owned fixture denial")):
    try:
        checker.check_mount("unused-transport-simulation-lock")
    except PermissionError as error:
        assert "root-owned fixture denial" in str(error)
    else:
        raise AssertionError("Cleanup denial was suppressed")
assert not (checker.artifacts / "mounted-cpu.json").exists()
assert checker.fixture.exists()
shutil.rmtree(checker.fixture)  # Only the host-owned fixture created by this replay.
records.append({"case":"cleanup-denial-prevents-acceptance", "accepted":True,
                "ownership_denial_simulated":True,"sudo_or_chown_executed":False,
                "mounted_acceptance_receipt_created":False})

assert shutil.which("docker") is None
record = {"accepted":True,"source":checks.source_record(root),"cases":records,
          "scope":"Actual local child imports and checker cleanup semantics; "
                  "Docker transport and root-ownership denial simulated",
          "actual_container_executed":False,"actual_root_owned_file_created":False}
(outputs / "regression.json").write_text(json.dumps(record,indent=2)+"\n")
print("Passed actual bytecode control/suppression, mount-checker cleanup and denial checks")
