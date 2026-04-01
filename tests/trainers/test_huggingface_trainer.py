from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from plato.config import Config


class DummyPadTokenizer:
    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.pad_token = None
        self.padding_side = "right"

    def __len__(self):
        return self.vocab_size

    def pad(self, features, padding=True, return_tensors=None):
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": []}
        if any("token_type_ids" in feature for feature in features):
            batch["token_type_ids"] = []

        for feature in features:
            pad_width = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_width)
            batch["attention_mask"].append(
                feature.get("attention_mask", [1] * len(feature["input_ids"]))
                + [0] * pad_width
            )
            if "token_type_ids" in batch:
                values = feature.get("token_type_ids", [0] * len(feature["input_ids"]))
                batch["token_type_ids"].append(values + [0] * pad_width)

        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
        }


class DummyHFModel(nn.Module):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 4)
        self.config = SimpleNamespace(use_cache=True)
        self.gradient_checkpointing_enabled = False
        self.resized_to = None

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch, seq = input_ids.shape
        vocab_size = self.embedding.num_embeddings
        logits = torch.zeros((batch, seq, vocab_size), dtype=torch.float32)
        return SimpleNamespace(loss=torch.tensor(0.0), logits=logits)

    def get_input_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, new_size: int):
        self.resized_to = new_size
        self.embedding = nn.Embedding(new_size, 4)
        return self.embedding

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

        def make_inputs_require_grads(*args, **kwargs):
            del args, kwargs
            return None

        self._require_grads_hook = make_inputs_require_grads


def test_huggingface_collate_wrapper_dynamically_pads_variable_length_labels():
    from plato.trainers.huggingface import HuggingFaceCollateWrapper

    wrapper = HuggingFaceCollateWrapper(tokenizer=DummyPadTokenizer())
    batch, labels = wrapper(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 3],
            },
            {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "labels": [-100, 5],
            },
        ]
    )

    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert labels.tolist() == [[-100, 2, 3], [-100, 5, -100]]


def test_huggingface_collate_wrapper_unwraps_nested_batch_encodings():
    from plato.trainers.huggingface import HuggingFaceCollateWrapper

    wrapper = HuggingFaceCollateWrapper(tokenizer=DummyPadTokenizer())
    batch, labels = wrapper(
        [
            {
                "input_ids": {
                    "input_ids": [1, 2, 3],
                    "attention_mask": [1, 1, 1],
                },
                "attention_mask": [1, 1],
                "labels": [-100, 2, 3],
            },
            {
                "input_ids": {
                    "input_ids": [4, 5],
                    "attention_mask": [1, 1],
                },
                "labels": [-100, 5],
            },
        ]
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert labels.tolist() == [[-100, 2, 3], [-100, 5, -100]]


def test_huggingface_collate_wrapper_requires_tokenizer_at_construction():
    from plato.trainers.huggingface import HuggingFaceCollateWrapper

    with pytest.raises(ValueError, match=r"tokenizer with pad\(\) support"):
        HuggingFaceCollateWrapper(None)


def test_huggingface_trainer_defaults_tokenizer_name_to_model_name(
    temp_config, monkeypatch
):
    from plato.trainers import huggingface as huggingface_trainer

    cfg = Config()
    cfg.trainer.type = "HuggingFace"
    cfg.trainer.model_name = "base-model"
    if hasattr(cfg.trainer, "tokenizer_name"):
        del cfg.trainer["tokenizer_name"]

    calls: list[str] = []
    monkeypatch.setattr(
        huggingface_trainer.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        huggingface_trainer.AutoTokenizer,
        "from_pretrained",
        lambda name, **kwargs: calls.append(name) or DummyPadTokenizer(),
    )

    trainer = huggingface_trainer.Trainer(model=DummyHFModel())

    assert calls == ["base-model"]
    assert trainer.tokenizer is not None


def test_huggingface_trainer_uses_tokenizer_override_and_resizes_embeddings(
    temp_config, monkeypatch
):
    from plato.trainers import huggingface as huggingface_trainer

    cfg = Config()
    cfg.trainer.type = "HuggingFace"
    cfg.trainer.model_name = "base-model"
    cfg.trainer.tokenizer_name = "chat-tokenizer"

    calls: list[str] = []
    monkeypatch.setattr(
        huggingface_trainer.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        huggingface_trainer.AutoTokenizer,
        "from_pretrained",
        lambda name, **kwargs: calls.append(name) or DummyPadTokenizer(vocab_size=12),
    )

    model = DummyHFModel(vocab_size=8)
    huggingface_trainer.Trainer(model=model)

    assert calls == ["chat-tokenizer"]
    assert model.resized_to == 12


def test_huggingface_trainer_applies_gradient_checkpointing_and_precision_flags(
    temp_config, monkeypatch
):
    from plato.trainers import huggingface as huggingface_trainer

    cfg = Config()
    cfg.trainer.type = "HuggingFace"
    cfg.trainer.model_name = "base-model"
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.bf16 = True
    cfg.trainer.fp16 = False

    monkeypatch.setattr(
        huggingface_trainer.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        huggingface_trainer.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyPadTokenizer(vocab_size=8),
    )

    model = DummyHFModel(vocab_size=8)
    trainer = huggingface_trainer.Trainer(model=model)

    assert trainer.training_args.gradient_checkpointing is True
    assert trainer.training_args.bf16 is True
    assert trainer.training_args.fp16 is False
    assert model.gradient_checkpointing_enabled is False
    assert model.config.use_cache is False

    # Gradient-checkpointing hooks must be installed lazily so the trainer can
    # still be pickled when multiprocessing uses the spawn start method.
    pickle.dumps(trainer)

    delegated: list[bool] = []

    def fake_super_train_model(self, config, trainset, sampler, **kwargs):
        del self, config, trainset, sampler, kwargs
        delegated.append(model.gradient_checkpointing_enabled)
        return "trained"

    monkeypatch.setattr(
        huggingface_trainer.ComposableTrainer,
        "train_model",
        fake_super_train_model,
    )

    result = trainer.train_model({"epochs": 1, "batch_size": 1}, [], [])

    assert result == "trained"
    assert delegated == [True]
    assert model.gradient_checkpointing_enabled is True
