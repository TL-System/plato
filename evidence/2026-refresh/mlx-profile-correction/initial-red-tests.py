"""Full production import graphs must respect the explicit native boundary."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _audit(tmp_path, arguments, *, mode="observe", directory=""):
    root = Path(__file__).resolve().parents[1]
    receipt = tmp_path / "imports.json"
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tests/mlx_import_audit_runner.py"),
            str(receipt),
            mode,
            *arguments,
        ],
        cwd=root / directory,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert receipt.exists(), output
    record = json.loads(receipt.read_text())
    assert record["production_conftest_loaded"], output
    assert record["exit_status"] == result.returncode, output
    return record, output


@pytest.mark.parametrize("target", [[], ["tests"], ["."], ["--pyargs", "tests"]])
def test_real_core_collection_does_not_import_or_probe_mlx(tmp_path, target):
    record, output = _audit(tmp_path, [*target, "--test-profile=base", "--collect-only"])
    assert record["exit_status"] == 0, output
    assert record["backend_spec_lookups"] == []
    assert record["backend_modules"] == []
    assert record["native_frontdoor_modules"] == []
    assert record["native_test_modules"] == []


def test_real_conftest_survives_backend_import_sentinel(tmp_path):
    record, output = _audit(
        tmp_path, ["tests", "--test-profile=base", "--collect-only"], mode="block"
    )
    assert record["exit_status"] == 0, output
    assert record["backend_spec_lookups"] == []
    assert record["backend_modules"] == []


@pytest.mark.parametrize(
    "selector,directory",
    [
        ("tests.mlx_native.test_phase3_controls", ""),
        ("mlx_native.test_phase3_controls", "tests"),
        ("test_phase3_controls", "tests/mlx_native"),
    ],
    ids=["canonical", "dotted-alias", "bare-alias"],
)
def test_real_unsupported_native_routing_never_initializes_backend(
    tmp_path, selector, directory
):
    record, output = _audit(tmp_path, ["--pyargs", selector], directory=directory)
    assert record["exit_status"] != 0
    assert "mlx-native selector unsupported" in output
    assert record["backend_spec_lookups"] == []
    assert record["backend_modules"] == []
    assert record["native_frontdoor_modules"] == []
    assert record["native_test_modules"] == []


def test_real_core_pyargs_preserves_import_isolation(tmp_path):
    record, output = _audit(tmp_path, ["--pyargs", "tests.test_config_loader"])
    assert record["exit_status"] == 0, output
    assert "15 passed" in output
    assert record["backend_spec_lookups"] == []
    assert record["backend_modules"] == []


def test_real_focused_native_backend_import_starts_in_preflight(tmp_path):
    record, output = _audit(
        tmp_path,
        [
            "tests/mlx_native/test_phase3_tree_contract.py::"
            "test_extracted_host_weights_own_snapshots",
            "--collect-only",
        ],
    )
    if record["platform"] == ["Darwin", "arm64"]:
        assert record["backend_spec_lookups"], output
        assert record["backend_spec_lookups"][0]["inside_preflight"]
        if record["mlx_distribution_version"] is not None:
            assert record["exit_status"] == 0, output
            assert record["backend_modules"]
            assert record["native_test_modules"]
            return
    assert record["exit_status"] != 0
    assert "mlx-native prerequisite failed" in output
    assert record["native_test_modules"] == []
