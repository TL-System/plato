"""Nanochat integration smoke checks (optional, require nanochat extras)."""

from __future__ import annotations

import importlib.util
import os

import pytest

from plato.config import Config, ConfigNode
from plato.utils.third_party import ensure_nanochat_importable

pytestmark = pytest.mark.integration

_RUSTBPE_AVAILABLE = importlib.util.find_spec("rustbpe") is not None


@pytest.mark.skipif(
    not _RUSTBPE_AVAILABLE,
    reason="Nanochat tokenizer tests require rustbpe extension (install nanochat extras).",
)
def test_nanochat_tokenizer_processor_round_trip(tmp_path):
    """Train a tiny tokenizer via rustbpe and encode a sample string."""
    pytest.importorskip(
        "tiktoken", reason="Nanochat tokenizer tests require tiktoken (nanochat extra)."
    )
    from plato.processors.nanochat_tokenizer import NanochatTokenizerProcessor

    corpus = ["hello nanochat", "minimal tokenizer exercise"]
    processor = NanochatTokenizerProcessor(
        train_corpus=corpus,
        vocab_size=300,
        prepend_bos=False,
    )
    encoded = processor.process("hello nanochat")
    assert isinstance(encoded, list)
    assert len(encoded) > 0


def test_nanochat_trainer_smoke(temp_config, tmp_path):
    """Run one training step with synthetic Nanochat data on CPU."""
    _ = pytest.importorskip(
        "torch", reason="Nanochat trainer smoke requires torch (nanochat extra)."
    )
    ensure_nanochat_importable()
    _ = pytest.importorskip(
        "nanochat",
        reason="Nanochat trainer smoke requires the nanochat package (nanochat extra).",
    )
    from plato.datasources.nanochat import DataSource as NanochatDataSource
    from plato.models.nanochat import Model as NanochatModel
    from plato.trainers.nanochat import Trainer as NanochatTrainer

    cfg = Config()
    cfg.trainer.type = "nanochat"
    cfg.trainer.model_name = "nanochat_smoke"
    cfg.trainer.batch_size = 2
    cfg.trainer.rounds = 1
    cfg.trainer.epochs = 1

    cfg.parameters.model = {
        "sequence_len": 16,
        "vocab_size": 512,
        "n_layer": 1,
        "n_head": 2,
        "n_kv_head": 2,
        "n_embd": 128,
    }

    datasource = NanochatDataSource(
        batch_size=cfg.trainer.batch_size,
        sequence_length=cfg.parameters.model["sequence_len"],
        mode="synthetic",
        max_train_batches=2,
        max_val_batches=1,
        device="cpu",
        vocab_size=cfg.parameters.model["vocab_size"],
        synthetic_seed=123,
    )

    model = NanochatModel.get(
        sequence_len=cfg.parameters.model["sequence_len"],
        vocab_size=cfg.parameters.model["vocab_size"],
        n_layer=cfg.parameters.model["n_layer"],
        n_head=cfg.parameters.model["n_head"],
        n_kv_head=cfg.parameters.model["n_kv_head"],
        n_embd=cfg.parameters.model["n_embd"],
        init_weights=True,
    )

    trainer = NanochatTrainer(model=model)
    trainset = datasource.get_train_set()
    elapsed = trainer.train(trainset, sampler=None)

    assert isinstance(elapsed, float)
    assert elapsed >= 0.0

    model_dir = Config().params["model_path"]
    checkpoint_name = f"{cfg.trainer.model_name}_{trainer.client_id}_{Config().params['run_id']}.safetensors"
    assert os.path.exists(os.path.join(model_dir, checkpoint_name))


def test_nanochat_trainer_selects_core_eval_strategy(temp_config, monkeypatch):
    """Ensure evaluation config triggers the CORE testing strategy."""
    _ = pytest.importorskip(
        "torch", reason="Nanochat trainer requires torch (nanochat extra)."
    )
    ensure_nanochat_importable()
    _ = pytest.importorskip(
        "nanochat",
        reason="Nanochat trainer requires the nanochat package (nanochat extra).",
    )
    from plato.models.nanochat import Model as NanochatModel
    from plato.trainers.nanochat import (
        NanochatCoreTestingStrategy,
    )
    from plato.trainers.nanochat import (
        Trainer as NanochatTrainer,
    )

    monkeypatch.setattr(
        "plato.evaluators.nanochat_core.run_core_evaluation",
        lambda *args, **kwargs: {
            "results": {},
            "centered_results": {},
            "core_metric": 0.0,
        },
    )

    cfg = Config()
    cfg.trainer.type = "nanochat"
    cfg.trainer.model_name = "nanochat_core"
    cfg.trainer.batch_size = 1
    cfg.trainer.rounds = 1
    cfg.trainer.epochs = 1
    cfg.parameters.model = {
        "sequence_len": 16,
        "vocab_size": 512,
        "n_layer": 1,
        "n_head": 2,
        "n_kv_head": 2,
        "n_embd": 128,
    }
    cfg.evaluation = ConfigNode.from_object(
        {
            "type": "nanochat_core",
            "max_per_task": 1,
        }
    )

    model = NanochatModel.get(
        sequence_len=cfg.parameters.model["sequence_len"],
        vocab_size=cfg.parameters.model["vocab_size"],
        n_layer=cfg.parameters.model["n_layer"],
        n_head=cfg.parameters.model["n_head"],
        n_kv_head=cfg.parameters.model["n_kv_head"],
        n_embd=cfg.parameters.model["n_embd"],
        init_weights=True,
    )

    trainer = NanochatTrainer(model=model)
    assert isinstance(trainer.testing_strategy, NanochatCoreTestingStrategy)
