from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from plato.config import Config, ConfigNode
from plato.evaluators import registry as evaluator_registry
from plato.evaluators.base import EvaluationInput
from plato.evaluators.runner import EVALUATION_PRIMARY_KEY, EVALUATION_RESULTS_KEY
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import TestingStrategy, TrainingContext


class DummyNanochatModel(nn.Module):
    """Minimal model stub compatible with Nanochat trainer/evaluator hooks."""

    def __init__(self, loss: float = 0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.loss = loss
        self.nanochat_tokenizer = object()

    def forward(self, inputs, targets=None, loss_reduction="sum"):
        del inputs, targets, loss_reduction
        return torch.tensor(self.loss, dtype=torch.float32, device=self.weight.device)


def _clear_evaluation_config() -> None:
    if hasattr(Config, "evaluation"):
        delattr(Config, "evaluation")


def test_nanochat_core_evaluator_reuses_precomputed_results_from_context(
    temp_config, monkeypatch
):
    from plato.evaluators.nanochat_core import NanochatCoreEvaluator

    core_results = {
        "results": {"task_a": 0.75},
        "centered_results": {"task_a": 0.8},
        "core_metric": 0.8,
    }
    context = TrainingContext()
    context.state["nanochat_core_results"] = core_results

    def _unexpected_run(*args, **kwargs):
        raise AssertionError("run_core_evaluation should not be called")

    monkeypatch.setattr(
        "plato.evaluators.nanochat_core.run_core_evaluation",
        _unexpected_run,
    )

    result = NanochatCoreEvaluator({"type": "nanochat_core"}).evaluate(
        EvaluationInput(
            model=DummyNanochatModel(),
            context=context,
            tokenizer=object(),
        )
    )

    assert result.evaluator == "nanochat_core"
    assert result.primary_metric == "core_metric"
    assert result.metrics == {"core_metric": 0.8}
    assert result.higher_is_better == {"core_metric": True}
    assert result.metadata == {
        "results": {"task_a": 0.75},
        "centered_results": {"task_a": 0.8},
    }


def test_nanochat_trainer_core_eval_populates_generic_evaluation_state(
    temp_config, monkeypatch
):
    from plato.datasources.nanochat import NanochatStreamingDataset
    from plato.trainers.nanochat import Trainer as NanochatTrainer

    core_results = {
        "results": {"task_a": 0.75},
        "centered_results": {"task_a": 0.8},
        "core_metric": 0.8,
    }

    monkeypatch.setattr(
        "plato.trainers.nanochat.ensure_nanochat_importable", lambda: None
    )
    monkeypatch.setattr(
        "plato.evaluators.nanochat_core.ensure_nanochat_importable", lambda: None
    )
    monkeypatch.setattr(
        "plato.trainers.nanochat.run_core_evaluation",
        lambda *args, **kwargs: core_results,
    )
    monkeypatch.setattr(
        "plato.evaluators.nanochat_core.run_core_evaluation",
        lambda *args, **kwargs: core_results,
    )

    cfg = Config()
    cfg.trainer.type = "nanochat"
    cfg.trainer.model_name = "nanochat_core"
    cfg.evaluation = ConfigNode.from_object(
        {
            "type": "nanochat_core",
            "max_per_task": 1,
        }
    )

    try:
        trainer = NanochatTrainer(model=DummyNanochatModel())
        testset = NanochatStreamingDataset(
            split="val",
            batch_size=1,
            sequence_length=4,
            mode="synthetic",
            base_dir=None,
            max_batches=1,
            tokenizer_threads=1,
            tokenizer_batch_size=1,
            device="cpu",
            vocab_size=32,
            synthetic_seed=123,
        )

        accuracy = trainer.test_model(
            config={"batch_size": 1},
            testset=testset,
            sampler=None,
        )

        assert isinstance(accuracy, float)
        assert trainer.context.state["nanochat_core_results"] == core_results
        assert trainer.context.state[EVALUATION_PRIMARY_KEY] == {
            "evaluator": "nanochat_core",
            "metric": "core_metric",
            "value": 0.8,
        }
        assert trainer.context.state[EVALUATION_RESULTS_KEY]["nanochat_core"] == {
            "evaluator": "nanochat_core",
            "primary_metric": "core_metric",
            "metrics": {"core_metric": 0.8},
            "higher_is_better": {"core_metric": True},
            "metadata": {
                "results": {"task_a": 0.75},
                "centered_results": {"task_a": 0.8},
            },
            "artifacts": {},
            "primary_value": 0.8,
        }
    finally:
        evaluator_registry.unregister("nanochat_core")
        _clear_evaluation_config()


def test_nanochat_trainer_does_not_register_core_evaluator_globally(
    temp_config, monkeypatch
):
    from plato.trainers.nanochat import Trainer as NanochatTrainer

    class ConstantTestingStrategy(TestingStrategy):
        def test_model(self, model, config, testset, sampler, context):
            return 0.5

    core_results = {
        "results": {},
        "centered_results": {},
        "core_metric": 0.8,
    }

    monkeypatch.setattr(
        "plato.trainers.nanochat.ensure_nanochat_importable", lambda: None
    )
    monkeypatch.setattr(
        "plato.evaluators.nanochat_core.ensure_nanochat_importable", lambda: None
    )
    monkeypatch.setattr(
        "plato.trainers.nanochat.run_core_evaluation",
        lambda *args, **kwargs: core_results,
    )
    monkeypatch.setattr(
        "plato.evaluators.nanochat_core.run_core_evaluation",
        lambda *args, **kwargs: core_results,
    )

    cfg = Config()
    cfg.trainer.type = "nanochat"
    cfg.trainer.model_name = "nanochat_core"
    cfg.evaluation = ConfigNode.from_object(
        {
            "type": "nanochat_core",
            "max_per_task": 1,
        }
    )

    registered_before = evaluator_registry.registered_names()
    _ = NanochatTrainer(model=DummyNanochatModel())

    assert evaluator_registry.registered_names() == registered_before

    unrelated_trainer = ComposableTrainer(
        model=nn.Linear(2, 1),
        testing_strategy=ConstantTestingStrategy(),
    )
    unrelated_trainer.test_model(config={"batch_size": 1}, testset=[], sampler=None)

    assert unrelated_trainer.context.state == {}
    _clear_evaluation_config()


def test_nanochat_trainer_refreshes_evaluation_mode_between_test_calls(
    temp_config, monkeypatch
):
    from plato.datasources.nanochat import NanochatStreamingDataset
    from plato.evaluators.base import EvaluationResult
    from plato.trainers.nanochat import (
        NanochatCoreTestingStrategy,
        NanochatTestingStrategy,
    )
    from plato.trainers.nanochat import (
        Trainer as NanochatTrainer,
    )

    class MockEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate(self, request):
            return EvaluationResult(
                evaluator="mock",
                primary_metric="mock_score",
                metrics={"mock_score": 0.2},
            )

    core_results = {
        "results": {"task_a": 0.75},
        "centered_results": {"task_a": 0.8},
        "core_metric": 0.8,
    }

    monkeypatch.setattr(
        "plato.trainers.nanochat.ensure_nanochat_importable", lambda: None
    )
    monkeypatch.setattr(
        "plato.evaluators.nanochat_core.ensure_nanochat_importable", lambda: None
    )
    monkeypatch.setattr(
        "plato.trainers.nanochat.run_core_evaluation",
        lambda *args, **kwargs: core_results,
    )
    monkeypatch.setattr(
        "plato.evaluators.nanochat_core.run_core_evaluation",
        lambda *args, **kwargs: core_results,
    )

    cfg = Config()
    cfg.trainer.type = "nanochat"
    cfg.trainer.model_name = "nanochat_core"
    cfg.evaluation = ConfigNode.from_object(
        {
            "type": "nanochat_core",
            "max_per_task": 1,
        }
    )
    evaluator_registry.register("mock", MockEvaluator)

    try:
        trainer = NanochatTrainer(model=DummyNanochatModel())
        testset = NanochatStreamingDataset(
            split="val",
            batch_size=1,
            sequence_length=4,
            mode="synthetic",
            base_dir=None,
            max_batches=1,
            tokenizer_threads=1,
            tokenizer_batch_size=1,
            device="cpu",
            vocab_size=32,
            synthetic_seed=123,
        )

        core_accuracy = trainer.test_model(
            config={"batch_size": 1},
            testset=testset,
            sampler=None,
        )

        assert core_accuracy == pytest.approx(0.8)
        assert isinstance(trainer.testing_strategy, NanochatCoreTestingStrategy)
        assert trainer.context.state["nanochat_core_results"] == core_results

        cfg.evaluation = ConfigNode.from_object({"type": "mock"})
        switched_accuracy = trainer.test_model(
            config={"batch_size": 1},
            testset=testset,
            sampler=None,
        )

        assert switched_accuracy == pytest.approx(0.0625)
        assert isinstance(trainer.testing_strategy, NanochatTestingStrategy)
        assert "nanochat_core_results" not in trainer.context.state
        assert trainer.context.state[EVALUATION_PRIMARY_KEY] == {
            "evaluator": "mock",
            "metric": "mock_score",
            "value": 0.2,
        }
        assert trainer.context.state[EVALUATION_RESULTS_KEY] == {
            "mock": {
                "evaluator": "mock",
                "primary_metric": "mock_score",
                "metrics": {"mock_score": 0.2},
                "higher_is_better": {},
                "metadata": {},
                "artifacts": {},
                "primary_value": 0.2,
            }
        }
    finally:
        evaluator_registry.unregister("mock")
        _clear_evaluation_config()
