"""Revision propagation and cache identity across the maintained HF loaders."""

from typing import Any

import pytest

from plato.config import Config
from plato.utils.huggingface import artifact_identity
from tests.integration.utils import configure_environment
from tests.test_utils.qwen3 import create_tiny_qwen3, reference_config


@pytest.mark.parametrize(
    "override,explicit,expected",
    [
        (None, None, "model-pin"),
        ("other-tokenizer", None, "main"),
        ("other-tokenizer", "tokenizer-pin", "tokenizer-pin"),
    ],
)
def test_tokenizer_revision_follows_only_same_model_repository(
    temp_config,
    override,
    explicit,
    expected,
):
    config = Config()
    config.trainer.model_name = "model"
    config.trainer.model_revision = "model-pin"
    config.trainer.tokenizer_name = override
    config.trainer.tokenizer_revision = explicit
    assert artifact_identity()["tokenizer_revision"] == expected


def test_all_hf_loaders_propagate_exact_revisions_with_real_local_artifacts(
    tmp_path,
    monkeypatch,
):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from plato.datasources.huggingface import DataSource as CorpusDataSource
    from plato.datasources.lora import DataSource as LoRADataSource
    from plato.models.huggingface import Model
    from plato.trainers.huggingface import Trainer

    directory = create_tiny_qwen3(tmp_path / "model")
    config = reference_config(directory)
    config["trainer"]["model_revision"] = "model-pin"
    config["trainer"]["tokenizer_revision"] = "tokenizer-pin"
    config["data"].update(block_size=16, preprocessing_num_proc=1)
    calls = {"config": [], "model": [], "tokenizer": []}
    for kind, loader in (
        ("config", AutoConfig),
        ("model", AutoModelForCausalLM),
        ("tokenizer", AutoTokenizer),
    ):
        original = loader.from_pretrained

        def spy(*args, _kind=kind, _original=original, **kwargs):
            calls[_kind].append(kwargs.copy())
            return _original(*args, **kwargs)

        monkeypatch.setattr(loader, "from_pretrained", spy)
    with configure_environment(config):
        model = Model.get()
        Trainer(model=model)
        LoRADataSource()
        CorpusDataSource()
    assert len(calls["model"]) == 1
    # LoRA's AutoTokenizer also resolves its own config at the tokenizer pin.
    assert sorted(call["revision"] for call in calls["config"]) == [
        "model-pin",
        "model-pin",
        "model-pin",
        "tokenizer-pin",
    ]
    assert len(calls["tokenizer"]) == 3
    for kind, revision in (("model", "model-pin"), ("tokenizer", "tokenizer-pin")):
        for call in calls[kind]:
            assert call["revision"] == revision
            assert call["trust_remote_code"] is False
            assert "use_auth_token" not in call


def test_pinned_model_loader_failure_never_retries_main(temp_config, monkeypatch):
    from plato.models.huggingface import AutoConfig, AutoModelForCausalLM, Model

    config = Config()
    config.trainer.model_name = "missing-pinned-model"
    config.trainer.model_revision = "missing-revision"
    calls = []

    def unavailable(name, **kwargs):
        calls.append((name, kwargs["revision"]))
        raise OSError("Pinned config is unavailable")

    def forbidden(*args, **kwargs):
        pytest.fail("Weight loading must not run after pinned config failure.")

    monkeypatch.setattr(AutoConfig, "from_pretrained", unavailable)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    with pytest.raises(OSError, match="Pinned config is unavailable"):
        Model.get()
    assert calls == [("missing-pinned-model", "missing-revision")]


def test_pinned_tokenizer_failure_never_retries_main(tmp_path, monkeypatch):
    from plato.datasources.lora import AutoTokenizer, DataSource

    directory = create_tiny_qwen3(tmp_path / "model")
    config = reference_config(directory)
    config["trainer"]["tokenizer_revision"] = "missing-revision"
    calls = []

    def unavailable(name, **kwargs):
        calls.append(kwargs["revision"])
        raise OSError("Pinned tokenizer is unavailable")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", unavailable)
    with (
        configure_environment(config),
        pytest.raises(OSError, match="Pinned tokenizer"),
    ):
        DataSource()
    assert calls == ["missing-revision"]


def test_dataset_cache_identity_includes_revision_local_path_and_content(tmp_path):
    from plato.datasources.huggingface import _dataset_cache_path

    file = tmp_path / "train.json"
    file.write_text('[{"text":"first"}]')
    kwargs: dict[str, Any] = dict(
        dataset_name="json",
        dataset_config=None,
        preprocessing_mode="corpus_lm",
        train_split="train",
        validation_split="validation",
        data_files={"train": str(file)},
    )
    first = _dataset_cache_path(str(tmp_path), dataset_revision="pin-a", **kwargs)
    assert first != _dataset_cache_path(
        str(tmp_path), dataset_revision="pin-b", **kwargs
    )
    file.write_text('[{"text":"second"}]')
    assert first != _dataset_cache_path(
        str(tmp_path), dataset_revision="pin-a", **kwargs
    )
    other = tmp_path / "other.json"
    other.write_text(file.read_text())
    changed_content = _dataset_cache_path(
        str(tmp_path), dataset_revision="pin-a", **kwargs
    )
    kwargs["data_files"] = {"train": str(other)}
    assert changed_content != _dataset_cache_path(
        str(tmp_path), dataset_revision="pin-a", **kwargs
    )


def test_pinned_dataset_does_not_accept_legacy_unpinned_cache(temp_config, monkeypatch):
    from plato.datasources import huggingface

    config = Config()
    config.data.dataset_name = "dataset"
    config.data.dataset_revision = "missing-pin"
    calls = []
    monkeypatch.setattr(
        huggingface.os.path, "exists", lambda path: path.endswith("dataset_None")
    )
    monkeypatch.setattr(
        huggingface,
        "load_from_disk",
        lambda path: pytest.fail("Unpinned legacy cache was accepted"),
    )

    def unavailable(name, **kwargs):
        calls.append(kwargs["revision"])
        raise OSError("Pinned dataset is unavailable")

    monkeypatch.setattr(huggingface, "load_dataset", unavailable)
    with pytest.raises(OSError, match="Pinned dataset"):
        huggingface.DataSource()
    assert calls == ["missing-pin"]
