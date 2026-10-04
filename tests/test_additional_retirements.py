"""Bounded archive preservation, ledger changes and installed-model replacements."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from plato.models import lenet5, registry, torchvision
from plato.trainers import optimizers
from tests.examples_phase4.common import _digest

REPO = Path(__file__).resolve().parents[1]
PLAN_SHA256 = "a49502dca7146305f37c94a404ff7735c7507d79d3968b5a813f09e304442716"
PLAN = json.loads(
    (REPO / "evidence/2026-refresh/plato-next-retirements-plan.json").read_text()
)
SCOPES = {scope["archive"]: scope for scope in PLAN["scopes"]}
POST_RETIREMENT_LEDGER_SHA256 = (
    "b654540d190c58e69e0a7d223b3fd59ab39d3d0f18b38fa7cffbe599f8e8f458"
)


def _verify_file(path, entry):
    mode = path.lstat().st_mode
    assert stat.S_ISREG(mode)
    assert bool(mode & 0o111) == (entry["git_mode"] == "100755")
    data = path.read_bytes()
    assert len(data) == entry["bytes"]
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == entry["git_blob"]
    )


@pytest.mark.parametrize("archive", SCOPES)
def test_archives_match_reviewed_immutable_inventory(archive):
    plan_path = REPO / "evidence/2026-refresh/plato-next-retirements-plan.json"
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == PLAN_SHA256
    scope = SCOPES[archive]
    folder = REPO / "archives/retired" / archive
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["plan_sha256"] == PLAN_SHA256
    assert (
        manifest["pre_retirement_plato_commit"] == PLAN["source_binding"]["base_commit"]
    )
    assert manifest["pre_retirement_plato_tree"] == PLAN["source_binding"]["base_tree"]
    expected = [
        {
            "original_path": entry["path"],
            "archive_path": entry["archive_path"],
            "disposition": entry["disposition"],
            "git_mode": entry["git_mode"],
            "git_blob": entry["git_blob"],
            "sha256": entry["sha256"],
            "bytes": entry["bytes"],
            "source_commit": PLAN["source_binding"]["base_commit"],
        }
        for entry in scope["source_inventory"]
    ]
    assert manifest["files"] == expected
    assert manifest["inventory_sha256"] == _digest(expected)
    assert {
        p.relative_to(REPO).as_posix()
        for p in (folder / "original").rglob("*")
        if p.is_file() or p.is_symlink()
    } == {entry["archive_path"] for entry in expected}
    for entry in expected:
        _verify_file(REPO / entry["archive_path"], entry)
        if entry["disposition"] == "moved":
            assert not (REPO / entry["original_path"]).exists()
    for entry in manifest["evidence"]:
        data = (REPO / entry["archive_path"]).read_bytes()
        assert len(data) == entry["bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    review = REPO / "evidence/2026-refresh/plato-next-retirements-plan-review.json"
    assert (
        manifest["plan_review_sha256"]
        == hashlib.sha256(review.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("damage", ["byte", "mode", "symlink"])
def test_archive_identity_verifier_rejects_damage(tmp_path, damage):
    entry = SCOPES["modality-samplers"]["source_inventory"][0]
    path = tmp_path / "source"
    path.write_bytes((REPO / entry["archive_path"]).read_bytes())
    path.chmod(0o644)
    _verify_file(path, entry)
    if damage == "mode":
        path.chmod(0o755)
    elif damage == "symlink":
        path.unlink()
        path.symlink_to(REPO / entry["archive_path"])
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(AssertionError):
        _verify_file(path, entry)


def test_historical_ledger_is_exact_reviewed_additional_delta():
    delta = PLAN["phase4_ledger"]
    prior_path = REPO / "evidence/2026-refresh/fei-post-retirement-ledger.json"
    assert (
        hashlib.sha256(prior_path.read_bytes()).hexdigest()
        == delta["base_ledger"]["sha256"]
    )
    prior = json.loads(prior_path.read_text())
    snapshot = REPO / "evidence/2026-refresh/additional-retirements-ledger.json"
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == (
        POST_RETIREMENT_LEDGER_SHA256
    )
    current = json.loads(snapshot.read_text())
    amendment = current["additional_retirement_amendment"]
    assert amendment["prior_ledger_sha256"] == delta["base_ledger"]["sha256"]
    assert amendment["plan_sha256"] == PLAN_SHA256
    removed = {r["path"] for r in delta["removed_source_rows"]}
    assert amendment["removed_paths"] == sorted(removed)
    assert amendment["path_renames"] == delta["path_renames"]
    expected = copy.deepcopy(prior)
    expected["source_inventory"] = [
        r for r in expected["source_inventory"] if r["path"] not in removed
    ]
    changes = {row["before"]["path"]: row for row in amendment["changed_source_rows"]}
    assert set(changes) == {r["path"] for r in delta["exact_config_rows_before"]} | set(
        delta["additional_source_hash_updates"]
    )
    for row in expected["source_inventory"]:
        if row["path"] in changes:
            change = changes[row["path"]]
            assert row == change["before"]
            after = change["after"]
            row.update(after)
    for after in delta["exact_config_rows_after"]:
        assert (
            after
            == changes[
                next(
                    before["path"]
                    for before in delta["exact_config_rows_before"]
                    if delta["path_renames"].get(before["path"], before["path"])
                    == after["path"]
                )
            ]["after"]
        )
    for task in expected["tasks"]:
        task["owned_paths"] = [
            delta["path_renames"].get(p, p)
            for p in task["owned_paths"]
            if p not in removed
        ]
        task["families"] = [
            delta["path_renames"].get(f, f)
            for f in task["families"]
            if f not in removed
        ]
        task["config_bindings"] = [
            b for b in task["config_bindings"] if b["config"] not in removed
        ]
        for binding in task["config_bindings"]:
            binding["config"] = delta["path_renames"].get(
                binding["config"], binding["config"]
            )
        revision = delta["inventory_revision_deltas"].get(task["task_id"])
        if revision:
            assert task["inventory_revision"] == revision["before"]
            task["inventory_revision"] = revision["after"]
    expected["path_map_sha256"] = _digest(expected["source_inventory"])
    expected["config_bindings_sha256"] = _digest(
        [b for t in expected["tasks"] for b in t["config_bindings"]]
    )
    expected["status"] = current["status"]
    expected["additional_retirement_amendment"] = amendment
    assert expected == current
    assert amendment["active_counts"] == {
        "tasks": 11,
        "paths": 390,
        "families": 57,
        "config_bindings": 131,
    }
    for archive, sha256 in amendment["archive_manifest_sha256"].items():
        assert (
            hashlib.sha256(
                (REPO / "archives/retired" / archive / "manifest.json").read_bytes()
            ).hexdigest()
            == sha256
        )


def test_live_ledger_keeps_retired_paths_out_and_retained_sources_current():
    current = json.loads((REPO / "tests/examples_phase4/cases.json").read_text())
    paths = {row["path"] for row in current["source_inventory"]}
    retired = {path for scope in SCOPES.values() for path in scope["moved"]}
    assert paths.isdisjoint(retired)
    assert set(PLAN["phase4_ledger"]["path_renames"].values()) <= paths
    for row in current["source_inventory"]:
        assert (
            row["sha256"]
            == hashlib.sha256((REPO / row["path"]).read_bytes()).hexdigest()
        ), row["path"]
    assert _digest(current["source_inventory"]) == current["path_map_sha256"]
    assert (
        _digest(
            [
                binding
                for task in current["tasks"]
                for binding in task["config_bindings"]
            ]
        )
        == current["config_bindings_sha256"]
    )


def test_retired_torch_hub_fails_before_factory(temp_config, monkeypatch):
    monkeypatch.setattr(
        torchvision.Model, "get", lambda **kw: pytest.fail("factory invoked")
    )
    with pytest.raises(ValueError, match="retired.*torch-hub/README.md"):
        registry.get(model_type="torch_hub", model_name="resnet18")
    assert importlib.util.find_spec("plato.models.torch_hub") is None
    for module in ["modality_iid", "modality_quantity_noniid"]:
        assert importlib.util.find_spec("plato.samplers." + module) is None


@pytest.mark.parametrize(
    "options, expected",
    [
        ({"pretrained": False}, {"weights": None}),
        ({"pretrained": True}, {"weights": "DEFAULT"}),
        ({"pretrained": True, "weights": None}, {"weights": None}),
        ({"weights": None, "num_classes": 7}, {"weights": None, "num_classes": 7}),
    ],
)
def test_installed_factory_preserves_weights_and_options(
    monkeypatch, options, expected
):
    sentinel = object()
    calls = []

    def factory(name, **kwargs):
        calls.append((name, kwargs))
        return sentinel

    monkeypatch.setattr(torchvision.models, "get_model", factory)
    assert torchvision.Model.get("resnet18", **options) is sentinel
    assert calls == [("resnet18", expected)]


def test_installed_torchvision_training_has_no_hub_activity(temp_config, monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *a, **kw: pytest.fail("Hub load"))
    monkeypatch.setattr(
        torch.hub, "load_state_dict_from_url", lambda *a, **kw: pytest.fail("download")
    )
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = registry.get(
            model_type="torchvision",
            model_name="resnet18",
            model_params={"weights": None, "num_classes": 7},
        )
        before = model.fc.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = F.cross_entropy(model(torch.randn(2, 3, 32, 32)), torch.tensor([0, 1]))
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        assert not torch.equal(before, model.fc.weight)
    finally:
        torch.set_num_threads(threads)


def test_adam_unlearning_recipe_trains_and_retired_optimizer_fails(monkeypatch):
    path = REPO / "examples/unlearning/fedunlearning/fedunlearning_MNIST_lenet5.toml"
    config = tomllib.loads(path.read_text())
    assert config["trainer"]["optimizer"] == "Adam"
    assert "create_graph" not in config["trainer"]
    assert "hessian_power" not in config["parameters"]["optimizer"]
    model = lenet5.Model()
    optimizer = optimizers.get(
        model,
        optimizer_name=config["trainer"]["optimizer"],
        optimizer_params=config["parameters"]["optimizer"],
    )
    before = next(model.parameters()).detach().clone()
    loss = F.nll_loss(model(torch.randn(2, 1, 28, 28)), torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()))
    with pytest.raises(ValueError, match="retired.*adahessian/README.md"):
        optimizers.get(model, optimizer_name="AdaHessian", optimizer_params={})


@pytest.mark.parametrize(
    "filename",
    [
        Path(p).name
        for scope in SCOPES.values()
        for p in scope["moved"]
        if p.endswith(".toml")
    ],
)
def test_missing_retired_recipe_has_archive_guidance(filename):
    env = dict(os.environ, config_file=filename)
    result = subprocess.run(
        [sys.executable, "-c", "from plato.config import Config; Config()"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "was retired" in result.stderr
    assert "archives/retired/" in result.stderr


def test_existing_custom_recipe_with_historical_name_still_loads(tmp_path):
    source = REPO / "examples/unlearning/fedunlearning/fedunlearning_MNIST_lenet5.toml"
    target = tmp_path / "fedunlearning_adahessian_MNIST_lenet5.toml"
    target.write_bytes(source.read_bytes())
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from plato.config import Config; assert Config().trainer.optimizer == 'Adam'",
        ],
        cwd=REPO,
        env=dict(os.environ, config_file=str(target)),
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
