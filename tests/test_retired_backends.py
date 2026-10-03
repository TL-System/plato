"""Retired integrations retain verifiable archives and reject old selections."""

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest
from torch import nn

from plato.config import Config, ConfigNode
from plato.datasources import registry as datasource_registry
from plato.evaluators import registry as evaluator_registry
from plato.evaluators.runner import run_configured_evaluation
from plato.models import registry as model_registry
from plato.trainers import registry as trainer_registry
from plato.trainers.strategies.base import TrainingContext

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model_type", ["nanochat", "smolvla"])
def test_retired_model_selection_fails_with_archive_guidance(temp_config, model_type):
    with pytest.raises(ValueError, match="retired.*archives/retired/"):
        model_registry.get(model_type=model_type, model_name=model_type)


@pytest.mark.parametrize("trainer_type", ["nanochat", "lerobot"])
def test_retired_trainer_selection_fails_with_archive_guidance(
    temp_config, trainer_type
):
    Config().trainer = Config().trainer._replace(type=trainer_type)
    with pytest.raises(ValueError, match="retired.*archives/retired/"):
        trainer_registry.get(model=nn.Linear(2, 1))


@pytest.mark.parametrize("datasource", ["Nanochat", "LeRobot"])
@pytest.mark.parametrize("input_shape", [False, True])
def test_retired_datasource_selection_fails_with_archive_guidance(
    temp_config, datasource, input_shape
):
    Config().data = Config().data._replace(datasource=datasource)
    with pytest.raises(ValueError, match="retired.*archives/retired/"):
        if input_shape:
            datasource_registry.get_input_shape()
        else:
            datasource_registry.get()


@pytest.mark.parametrize("allow_missing", [False, True])
def test_retired_evaluator_cannot_silently_disappear(temp_config, allow_missing):
    with pytest.raises(ValueError, match="retired.*archives/retired/nanochat"):
        evaluator_registry.get(
            {"type": "nanochat_core"}, allow_missing=allow_missing
        )


def test_retired_evaluator_cannot_run_via_override(temp_config):
    Config().evaluation = ConfigNode.from_object({"type": "nanochat_core"})

    class ObsoleteOverride:
        config = {"type": "nanochat_core"}

        def evaluate(self, request):
            pytest.fail("A retired evaluator must fail before evaluation.")

    with pytest.raises(ValueError, match="retired.*archives/retired/nanochat"):
        run_configured_evaluation(
            model=nn.Linear(2, 1),
            context=TrainingContext(),
            evaluator_override=ObsoleteOverride(),
        )


@pytest.mark.parametrize("backend", ["nanochat", "lerobot"])
def test_retirement_archive_preserves_original_bytes(backend):
    archive = REPOSITORY / "archives" / "retired" / backend
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["pre_retirement_plato_commit"] == (
        "9ee92c52307cd3ecdce0c7416708c504bdb52137"
    )
    assert manifest["files"]
    for entry in manifest["files"]:
        archived_file = REPOSITORY / entry["archive_path"]
        assert archived_file.is_relative_to(archive)
        data = archived_file.read_bytes()
        assert len(data) == entry["bytes"], entry["original_path"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        if entry["disposition"] == "moved":
            assert not (REPOSITORY / entry["original_path"]).exists()


def test_nanochat_upstream_archive_preserves_pinned_source_and_notice():
    archive = REPOSITORY / "archives" / "retired" / "nanochat"
    upstream = json.loads((archive / "manifest.json").read_text())["upstream"]
    assert upstream["gitlink_pin"] == "c75fe54aa7c1fa881701c246f9427bcbe4eee5a4"
    archive_path = REPOSITORY / upstream["archive_path"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == (
        upstream["archive_sha256"]
    )
    with tarfile.open(archive_path) as source:
        files = {member.name: member for member in source if member.isfile()}
        assert files.keys() == {entry["path"] for entry in upstream["files"]}
        for entry in upstream["files"]:
            stream = source.extractfile(files[entry["path"]])
            assert stream is not None
            data = stream.read()
            assert len(data) == entry["bytes"]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        license_stream = source.extractfile(files["LICENSE"])
        assert license_stream is not None
        notice = license_stream.read()
    assert notice == (REPOSITORY / upstream["license_path"]).read_bytes()
    assert b"Andrej Karpathy" in notice


@pytest.mark.parametrize(
    "module_name",
    [
        "plato.models.nanochat",
        "plato.models.smolvla",
        "plato.trainers.nanochat",
        "plato.trainers.lerobot",
        "plato.datasources.nanochat",
        "plato.datasources.lerobot",
        "plato.processors.nanochat_tokenizer",
        "plato.evaluators.nanochat_core",
        "plato.utils.third_party",
    ],
)
def test_retired_runtime_modules_are_unavailable(module_name):
    assert importlib.util.find_spec(module_name) is None
