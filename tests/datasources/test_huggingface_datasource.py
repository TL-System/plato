from __future__ import annotations

from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from plato.config import Config


class DummyTokenizer:
    model_max_length = 64

    def __call__(self, texts):
        return {"input_ids": [[1, 2, 3] for _ in texts]}


class DummyChatTokenizer:
    model_max_length = 256
    pad_token_id = 0

    role_ids = {"system": 71, "user": 72, "assistant": 73}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
    ):
        if not tokenize:
            return "".join(
                f"<{message['role']}>{message['content']}|"
                for message in messages
            )

        tokens = []
        for message in messages:
            tokens.append(self.role_ids[message["role"]])
            tokens.extend(100 + ord(char) for char in message["content"])
            tokens.append(0)
        if add_generation_prompt:
            tokens.append(99)
        return tokens


class DummyChatTokenizerWithBatchEncoding(DummyChatTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
    ):
        tokens = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if not tokenize:
            return tokens
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
        }


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
    trainset = datasource.require_trainset()
    testset = datasource.require_testset()

    assert datasource.train_split_name == "train"
    assert datasource.validation_split_name == "validation"
    assert trainset.num_rows == 1
    assert testset.num_rows == 1


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
    testset = datasource.require_testset()

    assert datasource.validation_split_name == "test"
    assert testset.num_rows == 1


def test_huggingface_datasource_loads_legacy_cache_path_when_present(
    temp_config, monkeypatch
):
    from plato.datasources import huggingface as huggingface_datasource

    cfg = Config()
    cfg.data.dataset_name = "legacy_ds"
    cfg.data.dataset_config = None
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

    legacy_path = (
        f"{Config().params['data_path']}/{cfg.data.dataset_name}_{cfg.data.dataset_config}"
    )
    loaded_paths: list[str] = []

    monkeypatch.setattr(
        huggingface_datasource.os.path,
        "exists",
        lambda path: path == legacy_path,
    )
    monkeypatch.setattr(
        huggingface_datasource,
        "load_from_disk",
        lambda path: loaded_paths.append(path) or dataset,
    )
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
    trainset = datasource.require_trainset()

    assert loaded_paths == [legacy_path]
    assert trainset.num_rows == 1


class LargeContextDummyTokenizer(DummyTokenizer):
    model_max_length = 4096



def test_huggingface_corpus_mode_keeps_legacy_default_block_size(
    temp_config, monkeypatch
):
    from plato.datasources import huggingface as huggingface_datasource

    cfg = Config()
    cfg.data.dataset_name = "dummy"
    cfg.data.text_field = "text"
    cfg.data.preprocessing_mode = "corpus_lm"
    cfg.data.train_split = "train"
    cfg.data.validation_split = "validation"
    if "block_size" in cfg.data:
        del cfg.data["block_size"]
    cfg.trainer.model_name = "dummy-model"

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["hello world"]}),
            "validation": Dataset.from_dict({"text": ["bye world"]}),
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
        lambda *args, **kwargs: LargeContextDummyTokenizer(),
    )

    datasource = huggingface_datasource.DataSource()

    assert datasource.block_size == 128


def test_chat_sft_preprocesses_messages_without_corpus_concatenation(
    temp_config, monkeypatch
):
    from plato.datasources import huggingface as huggingface_datasource

    cfg = Config()
    cfg.data.dataset_name = "dummy-chat"
    cfg.data.preprocessing_mode = "chat_sft"
    cfg.data.messages_field = "messages"
    cfg.data.label_strategy = "assistant_only"
    cfg.data.max_seq_length = 64
    cfg.data.train_split = "train"
    cfg.data.validation_split = "test"
    cfg.trainer.model_name = "dummy-chat-model"
    cfg.trainer.tokenizer_name = "dummy-chat-tokenizer"

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "messages": [
                        [
                            {"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a"},
                        ],
                        [
                            {"role": "user", "content": "v"},
                            {"role": "assistant", "content": "b"},
                        ],
                    ]
                }
            ),
            "test": Dataset.from_dict(
                {
                    "messages": [
                        [
                            {"role": "user", "content": "w"},
                            {"role": "assistant", "content": "c"},
                        ]
                    ]
                }
            ),
        }
    )

    monkeypatch.setattr(
        huggingface_datasource, "load_dataset", lambda *args, **kwargs: dataset
    )
    monkeypatch.setattr(
        huggingface_datasource, "load_from_disk", lambda *args, **kwargs: dataset
    )
    monkeypatch.setattr(huggingface_datasource.os.path, "exists", lambda *args: False)
    monkeypatch.setattr(
        huggingface_datasource.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        huggingface_datasource.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyChatTokenizer(),
    )

    datasource = huggingface_datasource.DataSource()
    trainset = datasource.require_trainset()

    assert trainset.num_rows == 2
    example = trainset[0]
    assert set(example) == {"input_ids", "attention_mask", "labels"}
    assert len(example["input_ids"]) == len(example["attention_mask"])
    assert len(example["input_ids"]) == len(example["labels"])


def test_chat_sft_masks_non_assistant_tokens_with_minus_100(temp_config, monkeypatch):
    from plato.datasources import huggingface as huggingface_datasource

    cfg = Config()
    cfg.data.dataset_name = "dummy-chat"
    cfg.data.preprocessing_mode = "chat_sft"
    cfg.data.messages_field = "messages"
    cfg.data.label_strategy = "assistant_only"
    cfg.data.max_seq_length = 64
    cfg.data.train_split = "train"
    cfg.data.validation_split = "test"
    cfg.trainer.model_name = "dummy-chat-model"

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "messages": [
                        [
                            {"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a"},
                        ]
                    ]
                }
            ),
            "test": Dataset.from_dict(
                {
                    "messages": [
                        [
                            {"role": "user", "content": "x"},
                            {"role": "assistant", "content": "y"},
                        ]
                    ]
                }
            ),
        }
    )

    monkeypatch.setattr(
        huggingface_datasource, "load_dataset", lambda *args, **kwargs: dataset
    )
    monkeypatch.setattr(
        huggingface_datasource, "load_from_disk", lambda *args, **kwargs: dataset
    )
    monkeypatch.setattr(huggingface_datasource.os.path, "exists", lambda *args: False)
    monkeypatch.setattr(
        huggingface_datasource.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        huggingface_datasource.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyChatTokenizer(),
    )

    datasource = huggingface_datasource.DataSource()
    example = datasource.require_trainset()[0]

    assert example["labels"][:3] == [-100, -100, -100]
    assert example["labels"][3:] == example["input_ids"][3:]


def test_chat_sft_honors_max_seq_length_and_messages_field(temp_config, monkeypatch):
    from plato.datasources import huggingface as huggingface_datasource

    cfg = Config()
    cfg.data.dataset_name = "dummy-chat"
    cfg.data.preprocessing_mode = "chat_sft"
    cfg.data.messages_field = "dialogue"
    cfg.data.label_strategy = "assistant_only"
    cfg.data.max_seq_length = 4
    cfg.data.train_split = "train"
    cfg.data.validation_split = "test"
    cfg.trainer.model_name = "dummy-chat-model"

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "dialogue": [
                        [
                            {"role": "user", "content": "long"},
                            {"role": "assistant", "content": "reply"},
                        ]
                    ]
                }
            ),
            "test": Dataset.from_dict(
                {
                    "dialogue": [
                        [
                            {"role": "user", "content": "hold"},
                            {"role": "assistant", "content": "out"},
                        ]
                    ]
                }
            ),
        }
    )

    monkeypatch.setattr(
        huggingface_datasource, "load_dataset", lambda *args, **kwargs: dataset
    )
    monkeypatch.setattr(
        huggingface_datasource, "load_from_disk", lambda *args, **kwargs: dataset
    )
    monkeypatch.setattr(huggingface_datasource.os.path, "exists", lambda *args: False)
    monkeypatch.setattr(
        huggingface_datasource.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        huggingface_datasource.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyChatTokenizer(),
    )

    datasource = huggingface_datasource.DataSource()
    example = datasource.require_trainset()[0]

    assert len(example["input_ids"]) == 4
    assert len(example["labels"]) == 4
    assert len(example["attention_mask"]) == 4


def test_chat_sft_normalizes_mapping_chat_template_outputs(temp_config, monkeypatch):
    from plato.datasources import huggingface as huggingface_datasource

    cfg = Config()
    cfg.data.dataset_name = "dummy-chat"
    cfg.data.preprocessing_mode = "chat_sft"
    cfg.data.messages_field = "messages"
    cfg.data.label_strategy = "assistant_only"
    cfg.data.max_seq_length = 16
    cfg.data.train_split = "train"
    cfg.data.validation_split = "test"
    cfg.trainer.model_name = "dummy-chat-model"

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "messages": [
                        [
                            {"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a"},
                        ]
                    ]
                }
            ),
            "test": Dataset.from_dict(
                {
                    "messages": [
                        [
                            {"role": "user", "content": "v"},
                            {"role": "assistant", "content": "b"},
                        ]
                    ]
                }
            ),
        }
    )

    monkeypatch.setattr(
        huggingface_datasource, "load_dataset", lambda *args, **kwargs: dataset
    )
    monkeypatch.setattr(
        huggingface_datasource, "load_from_disk", lambda *args, **kwargs: dataset
    )
    monkeypatch.setattr(huggingface_datasource.os.path, "exists", lambda *args: False)
    monkeypatch.setattr(
        huggingface_datasource.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        huggingface_datasource.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyChatTokenizerWithBatchEncoding(),
    )

    datasource = huggingface_datasource.DataSource()
    example = datasource.require_trainset()[0]

    assert isinstance(example["input_ids"], list)
    assert isinstance(example["attention_mask"], list)
    assert len(example["input_ids"]) == len(example["attention_mask"])
