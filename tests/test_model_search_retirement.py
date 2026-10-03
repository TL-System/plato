"""Immutable model-search archives and category-scoped runtime retirement."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
import torch

from plato.models import registry
from plato.utils.retired_backends import raise_if_retired

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "477f2f1bb52a1b5a58c1f9f2d30f640039f83a8e"
SOURCE_TREE = "82af1367f106218d0de98f46775c011e0afd930c"
# Independently reviewed plan inventories, including each original mapping/identity.
INVENTORIES = {
    "legacy-vit": "ba582b797ad70ce6b171400a679f42dea660e663ff387c012205b387fefef548",
    "fedtp": "386766c54c19579405e152d39fd95a8df936363e9993731d78192eda98c8ef2f",
    "pfedrlnas": "710638f97c93d37a2d9b0bd830a8f205167ef24321b19e16f6d43e7a2fcbca2c",
    "fedrlnas": "02af017a5aa172945152e299804f90402ce876521719fc1798b5fc27cfc5c02a",
    "anycostfl-vit": "9723d29a54af70a058ce062201930525f1ea35da3bb67a3071bf3adff5dd103f",
    "fedrolex-vit": "8271ed737b40c51bc4a63f46848f213790265cce390307fb2c0d4ff21971993c",
    "heterofl-mobilenetv3": "18ed7c4bb75184ce5b062050af37d6db738c00812f0e0076ac7a1f4fd5a6f183",
}
UPSTREAMS = {
    "plato/models/dvit": (
        "1ccb152cea43fcbc3cd517a45c12c65734f8ace3",
        352,
        "00e3a0cdbcb2e43955127f7b79c905668e98f0d635b411e030e1e3b7ac3236ca",
    ),
    "plato/models/t2tvit": (
        "0f63dc9558f4d192de926504dbddfa1b3f5db6ca",
        22,
        "7f9bfbc25c8d4e555580c46fbeeb4507d13ea71c0c0144bedf4751954c350aa8",
    ),
}
SOURCE_INVENTORY_SHA256 = (
    "f437fe8dd9b4c249f7357087a1d0ee6d4a204ec59b08268104af9084a4890846"
)


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _verify_file(path: Path, entry: dict[str, Any]) -> None:
    """Reject byte, Git-identity and executable-mode mutation independently."""
    mode = entry["git_mode"]
    actual_mode = path.lstat().st_mode
    if mode == "120000":
        assert stat.S_ISLNK(actual_mode), path
        data = os.fsencode(os.readlink(path))
    else:
        assert stat.S_ISREG(actual_mode), path
        data = path.read_bytes()
        assert bool(actual_mode & 0o111) == (mode == "100755"), path
    assert len(data) == entry["bytes"], path
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], path
    assert _git_blob(data) == entry["git_blob"], path


@pytest.mark.parametrize("family", INVENTORIES)
def test_archive_matches_reviewed_inventory_and_immutable_source(family: str) -> None:
    archive = REPOSITORY / "archives" / "retired" / family
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["pre_retirement_plato_commit"] == SOURCE_COMMIT
    assert manifest["pre_retirement_plato_tree"] == SOURCE_TREE
    files = manifest["files"]
    identities = sorted(
        (
            entry["original_path"],
            entry["archive_path"],
            entry["disposition"],
            entry["git_blob"],
            entry["git_mode"],
            entry["sha256"],
            entry["bytes"],
        )
        for entry in files
    )
    digest = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode())
    assert digest.hexdigest() == INVENTORIES[family]
    actual_paths = {
        path.relative_to(REPOSITORY).as_posix()
        for path in (archive / "original").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    assert actual_paths == {entry["archive_path"] for entry in files}
    # The receipt was verified against the immutable Git tree during capture.
    # Its pinned inventory digest also works in shallow CI checkouts.
    receipt = json.loads(
        (
            REPOSITORY / "evidence/2026-refresh/"
            "archive-shortlist-snapshot-verification.json"
        ).read_text()
    )
    assert receipt["pre_retirement_plato_commit"] == SOURCE_COMMIT
    assert receipt["pre_retirement_plato_tree"] == SOURCE_TREE
    source = receipt["immutable_source_inventory"]
    assert len(source) == 199
    digest = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    )
    assert digest.hexdigest() == SOURCE_INVENTORY_SHA256
    for entry in files:
        assert source[entry["original_path"]] == {
            key: entry[key] for key in ("git_mode", "git_blob", "sha256", "bytes")
        }
        target = REPOSITORY / entry["archive_path"]
        assert target.is_relative_to(archive / "original")
        _verify_file(target, entry)
        if entry["disposition"] == "moved":
            assert not (REPOSITORY / entry["original_path"]).exists()
    for notice in manifest["supplemental_licenses"]:
        data = (REPOSITORY / notice["archive_path"]).read_bytes()
        assert len(data) == notice["bytes"]
        assert hashlib.sha256(data).hexdigest() == notice["sha256"]
        assert notice["original_vendored_revision"] == "unknown"


def test_exact_pinned_upstream_snapshots_and_embedded_notices() -> None:
    manifest = json.loads(
        (REPOSITORY / "archives/retired/legacy-vit/manifest.json").read_text()
    )
    assert {u["original_path"] for u in manifest["upstreams"]} == UPSTREAMS.keys()
    for upstream in manifest["upstreams"]:
        pin, count, digest = UPSTREAMS[upstream["original_path"]]
        assert upstream["gitlink_pin"] == pin and len(upstream["files"]) == count
        snapshot = REPOSITORY / upstream["archive_path"]
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest
        assert upstream["archive_sha256"] == digest
        assert snapshot.stat().st_size == upstream["archive_bytes"]
        with tarfile.open(snapshot) as archive:
            members = {
                member.name: member
                for member in archive
                if member.isfile() or member.issym()
            }
            assert members.keys() == {entry["path"] for entry in upstream["files"]}
            for entry in upstream["files"]:
                member = members[entry["path"]]
                stream = archive.extractfile(member)
                assert stream is not None
                data = stream.read()
                assert _git_blob(data) == entry["git_blob"]
                assert hashlib.sha256(data).hexdigest() == entry["sha256"]
                assert len(data) == entry["bytes"]
                assert bool(member.mode & 0o111) == (entry["git_mode"] == "100755")
            for notice in upstream["notices"]:
                stream = archive.extractfile(members[notice["original_path"]])
                assert stream is not None
                assert (
                    stream.read() == (REPOSITORY / notice["archive_path"]).read_bytes()
                )


@pytest.mark.parametrize("mutation", ["byte", "mode"])
def test_verifier_rejects_temporary_copied_file_mutation(
    tmp_path: Path, mutation: str
) -> None:
    manifest = json.loads(
        (REPOSITORY / "archives/retired/legacy-vit/manifest.json").read_text()
    )
    entry = next(
        entry for entry in manifest["files"] if entry["disposition"] == "copied_context"
    )
    copied = tmp_path / "copied"
    copied.write_bytes((REPOSITORY / entry["archive_path"]).read_bytes())
    copied.chmod(0o644)
    _verify_file(copied, entry)
    if mutation == "byte":
        data = copied.read_bytes()
        copied.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    else:
        copied.chmod(0o755)
    with pytest.raises(AssertionError):
        _verify_file(copied, entry)


@pytest.mark.parametrize("name", ["google@vit-base-patch16-224", "deepvit", "t2tvit14"])
def test_retired_category_fails_before_factory_activity(
    temp_config: Any, monkeypatch: Any, name: str
) -> None:
    def forbidden(**kwargs: Any) -> None:
        pytest.fail("Retired category reached another factory.")

    monkeypatch.setattr(registry.huggingface.Model, "get", forbidden)
    monkeypatch.setattr(registry.torch_hub.Model, "get", forbidden)
    with pytest.raises(
        ValueError, match="retired.*archives/retired/legacy-vit/README.md"
    ):
        registry.get(model_type="vit", model_name=name)
    with pytest.raises(ValueError, match="No such model: unknown_model"):
        registry.get(model_type="unknown", model_name="unknown_model")


@pytest.mark.parametrize("name", ["vit_b_16", "mobilenet_v3_small"])
def test_generic_torchvision_category_still_delegates(
    temp_config: Any, monkeypatch: Any, name: str
) -> None:
    calls = []
    sentinel = object()

    def factory(**kwargs: Any) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(registry.torch_hub.Model, "get", factory)
    assert registry.get(model_type="torch_hub", model_name=name) is sentinel
    assert calls == [{"model_name": name}]
    raise_if_retired("vit", category="datasource")
    raise_if_retired("vit", category="trainer")


def test_independent_torchvision_mobilenet_backward() -> None:
    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(weights=None)
    model.eval()
    outputs = model(torch.zeros(2, 3, 32, 32))
    loss = outputs.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_independent_huggingface_category_delegates(
    temp_config: Any, monkeypatch: Any
) -> None:
    sentinel = object()
    calls = []

    def factory(**kwargs: Any) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(registry.huggingface.Model, "get", factory)
    assert (
        registry.get(model_type="huggingface", model_name="google@vit-base-patch16-224")
        is sentinel
    )
    assert calls == [{"model_name": "google@vit-base-patch16-224"}]


@pytest.mark.parametrize("name", ["vit", "dvit", "t2tvit"])
def test_retired_runtime_modules_absent(name: str) -> None:
    assert importlib.util.find_spec("plato.models." + name) is None


def test_registry_imports_in_fresh_process_without_submodules() -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-c", "import plato.models.registry"],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
