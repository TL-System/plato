from __future__ import annotations

import pytest
import torch.nn as nn

from plato.config import Config, ConfigNode
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import TestingStrategy


def _clear_evaluation_config() -> None:
    if hasattr(Config, "evaluation"):
        delattr(Config, "evaluation")


class ConstantTestingStrategy(TestingStrategy):
    def __init__(self, value: float):
        self.value = value

    def test_model(self, model, config, testset, sampler, context):
        return self.value


class MockEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, request):
        from plato.evaluators.base import EvaluationResult

        assert request.local_metric == 0.5
        return EvaluationResult(
            evaluator="mock",
            primary_metric="mock_score",
            metrics={"mock_score": 0.8, "aux_metric": 0.2},
            higher_is_better={"mock_score": True, "aux_metric": False},
            metadata={"source": "unit-test"},
        )


def test_evaluator_registry_resolves_registered_evaluator(temp_config):
    from plato.evaluators import registry as evaluator_registry

    _clear_evaluation_config()
    Config().evaluation = ConfigNode.from_object({"type": "mock"})
    evaluator_registry.register("mock", MockEvaluator)

    try:
        evaluator = evaluator_registry.get()
        assert isinstance(evaluator, MockEvaluator)
        assert evaluator.config["type"] == "mock"
    finally:
        evaluator_registry.unregister("mock")
        _clear_evaluation_config()


def test_evaluator_registry_rejects_unknown_type(temp_config):
    from plato.evaluators import registry as evaluator_registry

    _clear_evaluation_config()
    Config().evaluation = ConfigNode.from_object({"type": "missing"})

    with pytest.raises(ValueError, match="No such evaluator"):
        evaluator_registry.get()

    _clear_evaluation_config()


def test_composable_trainer_runs_registered_evaluator_and_stores_results(temp_config):
    from plato.evaluators import registry as evaluator_registry
    from plato.evaluators.runner import (
        EVALUATION_PRIMARY_KEY,
        EVALUATION_RESULTS_KEY,
    )

    _clear_evaluation_config()
    Config().evaluation = ConfigNode.from_object({"type": "mock"})
    evaluator_registry.register("mock", MockEvaluator)

    try:
        trainer = ComposableTrainer(
            model=nn.Linear(2, 1),
            testing_strategy=ConstantTestingStrategy(0.5),
        )

        accuracy = trainer.test_model(config={"batch_size": 1}, testset=[], sampler=None)

        assert accuracy == 0.5
        assert trainer.accuracy == 0.5
        assert trainer.context.state[EVALUATION_PRIMARY_KEY] == {
            "evaluator": "mock",
            "metric": "mock_score",
            "value": 0.8,
        }
        assert trainer.context.state[EVALUATION_RESULTS_KEY]["mock"]["metrics"] == {
            "mock_score": 0.8,
            "aux_metric": 0.2,
        }
    finally:
        evaluator_registry.unregister("mock")
        _clear_evaluation_config()


@pytest.mark.parametrize(
    "evaluation_config",
    [
        None,
        {"type": "nanochat_core"},
    ],
)
def test_composable_trainer_without_evaluator_keeps_legacy_test_behavior(
    temp_config, evaluation_config
):
    from plato.evaluators.runner import (
        EVALUATION_PRIMARY_KEY,
        EVALUATION_RESULTS_KEY,
    )

    _clear_evaluation_config()
    if evaluation_config is not None:
        Config().evaluation = ConfigNode.from_object(evaluation_config)
    trainer = ComposableTrainer(
        model=nn.Linear(2, 1),
        testing_strategy=ConstantTestingStrategy(0.5),
    )

    accuracy = trainer.test_model(config={"batch_size": 1}, testset=[], sampler=None)

    assert accuracy == 0.5
    assert trainer.accuracy == 0.5
    assert EVALUATION_RESULTS_KEY not in trainer.context.state
    assert EVALUATION_PRIMARY_KEY not in trainer.context.state
    _clear_evaluation_config()
