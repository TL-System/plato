"""Immutable FEI preservation and exact active qualification retirement."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import pytest

from tests.examples_phase4.common import _digest
from tests.test_mlx_profile_contract import policy, profile_source

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "archives/retired/fei"
SOURCE_COMMIT = "3d4e08b26972e114d87ef39795e01f4dfa554b94"
SOURCE_TREE = "16602a4bd057c30ec2bf195b926d6f2d405774dd"
INTEGRATION_COMMIT = "e81248dae1844fb5bc66a8d880c9a17b485606ac"
INVENTORY_SHA256 = "4f88e8e1ba6493448f7b34ec3f08c60615e5f6ab52feff7b2fec8e27f1d5875f"
PRIOR_LEDGER_SHA256 = "3e2aaca8daf3343ec4e7d31960185606f86992653b0772b0255adca47834a44d"
PLAN_SHA256 = "2d46ccccaa35c1753c44c1974d226405d49d04cdd3178812f816879320ed6667"
PLAN_REVIEW_SHA256 = "2bc7f5a374040880b18f29026881dc91b822d3815cf09ea9b7d73da74cebe303"
REJECTED_REVIEW_SHA256 = (
    "80094b3cec8272a8f44456ef2df4244579f9076990885e299b83b09d4e1f11d1"
)
RAW_EVIDENCE_SHA256 = "3061941926be598c454532b62b909fbb5bc3ba17bc780004767514967d6db44e"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _verify_file(path: Path, entry: dict[str, Any]) -> None:
    mode = path.lstat().st_mode
    assert stat.S_ISREG(mode), path
    assert bool(mode & 0o111) == (entry["git_mode"] == "100755"), path
    data = path.read_bytes()
    assert len(data) == entry["bytes"], path
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], path
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    assert hashlib.sha1(blob).hexdigest() == entry["git_blob"], path


def test_archive_matches_independently_reviewed_source_inventory() -> None:
    manifest = _read(ARCHIVE / "manifest.json")
    assert manifest["pre_retirement_plato_commit"] == SOURCE_COMMIT
    assert manifest["pre_retirement_plato_tree"] == SOURCE_TREE
    assert manifest["integration_commit"] == INTEGRATION_COMMIT
    assert manifest["plan_sha256"] == PLAN_SHA256
    assert manifest["plan_review_sha256"] == PLAN_REVIEW_SHA256
    files = manifest["files"]
    assert _digest(files) == manifest["inventory_sha256"] == INVENTORY_SHA256
    assert len(files) == 10
    assert sum(e["disposition"] == "moved" for e in files) == 6
    actual = {
        p.relative_to(REPO).as_posix()
        for p in (ARCHIVE / "original").rglob("*")
        if p.is_file() or p.is_symlink()
    }
    assert actual == {e["archive_path"] for e in files}
    for entry in files:
        path = REPO / entry["archive_path"]
        assert path.is_relative_to(ARCHIVE / "original")
        _verify_file(path, entry)
        expected_commit = (
            INTEGRATION_COMMIT
            if entry["original_path"] == "tests/examples_phase4/cases.json"
            else SOURCE_COMMIT
        )
        assert entry["source_commit"] == expected_commit
        if entry["disposition"] == "moved":
            assert not (REPO / entry["original_path"]).exists()
    assert not list(ARCHIVE.rglob("*.pdf"))
    assert not list(ARCHIVE.rglob("*.tgz"))
    assert manifest["external_raw_evidence"]["sha256"] == RAW_EVIDENCE_SHA256
    assert manifest["external_raw_evidence"]["distributed"] is False
    evidence = manifest["evidence"]
    assert [e["sha256"] for e in evidence] == [
        PLAN_SHA256,
        PLAN_REVIEW_SHA256,
        REJECTED_REVIEW_SHA256,
    ]
    for entry in evidence:
        data = (REPO / entry["archive_path"]).read_bytes()
        assert len(data) == entry["bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("mutation", ["byte", "mode"])
def test_archive_verifier_rejects_mutation(tmp_path: Path, mutation: str) -> None:
    entry = _read(ARCHIVE / "manifest.json")["files"][0]
    target = tmp_path / "source"
    target.write_bytes((REPO / entry["archive_path"]).read_bytes())
    target.chmod(0o644)
    _verify_file(target, entry)
    if mutation == "mode":
        target.chmod(0o755)
    else:
        data = target.read_bytes()
        target.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(AssertionError):
        _verify_file(target, entry)


def test_active_ledger_is_exact_reviewed_retirement_delta() -> None:
    prior_path = ARCHIVE / "original/tests/examples_phase4/cases.json"
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == PRIOR_LEDGER_SHA256
    prior = _read(prior_path)
    current = _read(REPO / "tests/examples_phase4/cases.json")
    retired = next(r for r in prior["tasks"] if r["task_id"] == "P4-fei")
    assert retired["state"] == "frozen" and len(retired["cases"]) == 7
    removed = set(retired["owned_paths"])
    assert len(removed) == 6 and len(retired["config_bindings"]) == 2
    amendment = current["retirement_amendment"]
    assert amendment["removed_paths"] == retired["owned_paths"]
    assert amendment["removed_config_bindings"] == retired["config_bindings"]
    assert amendment["prior_ledger_sha256"] == PRIOR_LEDGER_SHA256
    assert amendment["plan_sha256"] == PLAN_SHA256
    assert amendment["plan_review_sha256"] == PLAN_REVIEW_SHA256
    assert amendment["prior_path_map_sha256"] == prior["path_map_sha256"]
    assert amendment["prior_config_bindings_sha256"] == prior["config_bindings_sha256"]
    assert (
        amendment["archive_manifest_sha256"]
        == hashlib.sha256(
            (REPO / amendment["archive_manifest"]).read_bytes()
        ).hexdigest()
    )
    expected = copy.deepcopy(prior)
    expected["tasks"] = [r for r in prior["tasks"] if r["task_id"] != "P4-fei"]
    expected["source_inventory"] = [
        r for r in prior["source_inventory"] if r["path"] not in removed
    ]
    expected["path_map_sha256"] = _digest(expected["source_inventory"])
    bindings = [b for r in expected["tasks"] for b in r["config_bindings"]]
    expected["config_bindings_sha256"] = _digest(bindings)
    expected["status"] = current["status"]
    expected["retirement_amendment"] = amendment
    assert current == expected
    counts = {
        "tasks": len(current["tasks"]),
        "paths": len(current["source_inventory"]),
        "families": len({f for r in current["tasks"] for f in r["families"]}),
        "config_bindings": len(bindings),
    }
    assert (
        counts
        == amendment["active_counts"]
        == {
            "tasks": 11,
            "paths": 393,
            "families": 57,
            "config_bindings": 134,
        }
    )
    assert "fei" not in {f for r in current["tasks"] for f in r["families"]}


def test_retired_owner_cannot_qualify(policy: Any) -> None:
    with pytest.raises(pytest.UsageError, match="module has no task owner"):
        policy._load_phase4_ledger(
            REPO / "tests/examples_phase4/cases.json",
            "tests/examples_phase4/test_fei.py",
        )


@pytest.mark.parametrize("damage", ["missing-owner", "extra-owner", "duplicate-owner"])
def test_owner_inventory_damage_cannot_qualify(
    policy: Any, tmp_path: Path, damage: str
) -> None:
    ledger = _read(REPO / "tests/examples_phase4/cases.json")
    if damage == "missing-owner":
        ledger["tasks"].pop()
    elif damage == "extra-owner":
        prior = _read(ARCHIVE / "original/tests/examples_phase4/cases.json")
        ledger["tasks"].append(
            next(r for r in prior["tasks"] if r["task_id"] == "P4-fei")
        )
    else:
        ledger["tasks"][-1] = copy.deepcopy(ledger["tasks"][0])
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(ledger))
    message = "duplicate task ownership" if damage == "duplicate-owner" else "eleven"
    with pytest.raises(pytest.UsageError, match=message):
        policy._load_phase4_ledger(path, "tests/examples_phase4/test_core.py")
