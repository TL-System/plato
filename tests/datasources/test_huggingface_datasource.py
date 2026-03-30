from __future__ import annotations

from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from plato.config import Config


class DummyTokenizer:
    model_max_length = 64

    def __call__(self, texts):
        return {"input_ids": [[1, 2, 3] for _ in texts]}


def test_resolve_validation_split_falls_back_to_test_when_missing(temp_config):
    from plato.datasources.huggingface import _resolve_split_name

    dataset = {"train": object(), "test": object()}

    assert _resolve_split_name(dataset, "validation", fallback="test") == "test"


def test_dataset_cache_path_sanitizes_namespaced_inputs(temp_config):
    from plato.datasources.huggingface import _dataset_cache_path

    path = _dataset_cache_path(
        "/tmp/data",
        dataset_name="HuggingFaceTB/smol-smoltalk",
        dataset_config=None,
        preprocessing_mode="chat_sft",
        train_split="train",
        validation_split="test",
    )

    assert path.startswith("/tmp/data/")
    assert "HuggingFaceTB/smol-smoltalk" not in path
    assert "chat_sft" in path
    assert "test" in path


def test_preprocess_split_dispatches_by_mode(temp_config):
    from plato.datasources.huggingface import DataSource

    datasource = DataSource.__new__(DataSource)
    datasource.preprocessing_mode = "corpus_lm"
    datasource.preprocess_corpus_lm = lambda split: "corpus-result"
    datasource.preprocess_chat_sft = lambda split: "chat-result"

    assert datasource.preprocess_split(object()) == "corpus-result"

    datasource.preprocessing_mode = "chat_sft"
    assert datasource.preprocess_split(object()) == "chat-result"


def test_huggingface_datasource_keeps_validation_split_for_corpus_mode(
    temp_config, monkeypatch
):
    from plato.datasources import huggingface as huggingface_datasource

    cfg = Config()
    cfg.data.dataset_name = "dummy"
    cfg.data.text_field = "text"
    cfg.data.preprocessing_mode = "corpus_lm"
    cfg.data.train_split = "train"
    cfg.data.validation_split = "validation"
    cfg.trainer.model_name = "dummy-model"

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["hello"]}),
            "validation": Dataset.from_dict({"text": ["world"]}),
        }
    )

    monkeypatch.setattr(huggingface_datasource, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(huggingface_datasource, "load_from_disk", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(huggingface_datasource.os.path, "exists", lambda *args: False)
    monkeypatch.setattr(
        huggingface_datasource.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        huggingface_datasource.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(
        huggingface_datasource.DataSource,
        "preprocess_corpus_lm",
        lambda self, split: split,
    )

    datasource = huggingface_datasource.DataSource()

    assert datasource.train_split_name == "train"
    assert datasource.validation_split_name == "validation"
    assert datasource.trainset.num_rows == 1
    assert datasource.testset.num_rows == 1


def test_huggingface_datasource_falls_back_to_test_split(temp_config, monkeypatch):
    from plato.datasources import huggingface as huggingface_datasource

    cfg = Config()
    cfg.data.dataset_name = "dummy"
    cfg.data.text_field = "text"
    cfg.data.preprocessing_mode = "corpus_lm"
    cfg.data.train_split = "train"
    cfg.data.validation_split = "validation"
    cfg.trainer.model_name = "dummy-model"

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["hello"]}),
            "test": Dataset.from_dict({"text": ["world"]}),
        }
    )

    monkeypatch.setattr(huggingface_datasource, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(huggingface_datasource, "load_from_disk", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(huggingface_datasource.os.path, "exists", lambda *args: False)
    monkeypatch.setattr(
        huggingface_datasource.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        huggingface_datasource.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(
        huggingface_datasource.DataSource,
        "preprocess_corpus_lm",
        lambda self, split: split,
    )

    datasource = huggingface_datasource.DataSource()

    assert datasource.validation_split_name == "test"
    assert datasource.testset.num_rows == 1


def test_chat_mode_scaffold_is_explicit_not_implicit(temp_config):
    from plato.datasources.huggingface import DataSource

    datasource = DataSource.__new__(DataSource)
    datasource.preprocessing_mode = "chat_sft"

    with pytest.raises(NotImplementedError, match="chat_sft"):
        DataSource.preprocess_chat_sft(datasource, object())
