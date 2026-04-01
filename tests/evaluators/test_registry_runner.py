from __future__ import annotations

import pytest
import torch
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


class FailingEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, request):
        del request
        raise RuntimeError("boom")


class GradDisablingEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, request):
        from plato.evaluators.base import EvaluationResult

        del request
        torch.set_grad_enabled(False)
        return EvaluationResult(
            evaluator="grad-disabler",
            primary_metric="metric",
            metrics={"metric": 1.0},
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


def test_composable_trainer_replaces_stale_evaluator_payloads(temp_config):
    from plato.evaluators import registry as evaluator_registry
    from plato.evaluators.base import EvaluationResult
    from plato.evaluators.runner import EVALUATION_RESULTS_KEY
    from plato.servers import evaluation_logging

    class EvaluatorA:
        def __init__(self, config):
            self.config = config

        def evaluate(self, request):
            return EvaluationResult(
                evaluator="eval_a",
                primary_metric="metric_a",
                metrics={"metric_a": 0.1},
            )

    class EvaluatorB:
        def __init__(self, config):
            self.config = config

        def evaluate(self, request):
            return EvaluationResult(
                evaluator="eval_b",
                primary_metric="metric_b",
                metrics={"metric_b": 0.2},
            )

    _clear_evaluation_config()
    evaluator_registry.register("eval_a", EvaluatorA)
    evaluator_registry.register("eval_b", EvaluatorB)

    try:
        trainer = ComposableTrainer(
            model=nn.Linear(2, 1),
            testing_strategy=ConstantTestingStrategy(0.5),
        )

        Config().evaluation = ConfigNode.from_object({"type": "eval_a"})
        trainer.test_model(config={"batch_size": 1}, testset=[], sampler=None)

        Config().evaluation = ConfigNode.from_object({"type": "eval_b"})
        trainer.test_model(config={"batch_size": 1}, testset=[], sampler=None)

        assert trainer.context.state[EVALUATION_RESULTS_KEY] == {
            "eval_b": {
                "evaluator": "eval_b",
                "primary_metric": "metric_b",
                "metrics": {"metric_b": 0.2},
                "higher_is_better": {},
                "metadata": {},
                "artifacts": {},
                "primary_value": 0.2,
            }
        }
        assert evaluation_logging.extract_logged_items(trainer) == {
            "evaluation_primary_value": 0.2,
            "evaluation_metric_b": 0.2,
        }
    finally:
        evaluator_registry.unregister("eval_a")
        evaluator_registry.unregister("eval_b")
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


def test_composable_trainer_raises_for_unknown_evaluator_config(temp_config):
    _clear_evaluation_config()
    Config().evaluation = ConfigNode.from_object({"type": "unknown-evaluator"})
    trainer = ComposableTrainer(
        model=nn.Linear(2, 1),
        testing_strategy=ConstantTestingStrategy(0.5),
    )

    with pytest.raises(ValueError, match="No such evaluator"):
        trainer.test_model(config={"batch_size": 1}, testset=[], sampler=None)

    _clear_evaluation_config()


def test_composable_trainer_tolerates_evaluator_runtime_failure_by_default(
    temp_config,
):
    from plato.evaluators import registry as evaluator_registry
    from plato.evaluators.runner import (
        EVALUATION_PRIMARY_KEY,
        EVALUATION_RESULTS_KEY,
    )

    _clear_evaluation_config()
    Config().evaluation = ConfigNode.from_object({"type": "failing"})
    evaluator_registry.register("failing", FailingEvaluator)

    try:
        trainer = ComposableTrainer(
            model=nn.Linear(2, 1),
            testing_strategy=ConstantTestingStrategy(0.5),
        )

        accuracy = trainer.test_model(config={"batch_size": 1}, testset=[], sampler=None)

        assert accuracy == 0.5
        assert trainer.accuracy == 0.5
        assert EVALUATION_RESULTS_KEY not in trainer.context.state
        assert EVALUATION_PRIMARY_KEY not in trainer.context.state
    finally:
        evaluator_registry.unregister("failing")
        _clear_evaluation_config()


def test_composable_trainer_can_make_evaluator_failures_fatal(temp_config):
    from plato.evaluators import registry as evaluator_registry

    _clear_evaluation_config()
    Config().evaluation = ConfigNode.from_object(
        {"type": "failing", "fail_on_error": True}
    )
    evaluator_registry.register("failing", FailingEvaluator)

    try:
        trainer = ComposableTrainer(
            model=nn.Linear(2, 1),
            testing_strategy=ConstantTestingStrategy(0.5),
        )

        with pytest.raises(RuntimeError, match="boom"):
            trainer.test_model(config={"batch_size": 1}, testset=[], sampler=None)
    finally:
        evaluator_registry.unregister("failing")
        _clear_evaluation_config()


def test_composable_trainer_restores_grad_mode_after_evaluator_side_effect(
    temp_config,
):
    from plato.evaluators import registry as evaluator_registry

    _clear_evaluation_config()
    Config().evaluation = ConfigNode.from_object({"type": "grad-disabler"})
    evaluator_registry.register("grad-disabler", GradDisablingEvaluator)

    try:
        trainer = ComposableTrainer(
            model=nn.Linear(2, 1),
            testing_strategy=ConstantTestingStrategy(0.5),
        )

        assert torch.is_grad_enabled() is True
        accuracy = trainer.test_model(config={"batch_size": 1}, testset=[], sampler=None)

        assert accuracy == 0.5
        assert torch.is_grad_enabled() is True
    finally:
        evaluator_registry.unregister("grad-disabler")
        _clear_evaluation_config()
