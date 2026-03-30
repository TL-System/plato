from __future__ import annotations

from pathlib import Path

import pytest

from plato.config import Config, ConfigNode
from plato.evaluators.base import EvaluationInput


def _clear_evaluation_config() -> None:
    if hasattr(Config, "evaluation"):
        delattr(Config, "evaluation")


def test_lighteval_registry_resolves_without_importing_optional_backend(temp_config):
    from plato.evaluators import registry as evaluator_registry
    from plato.evaluators.lighteval import LightevalEvaluator

    _clear_evaluation_config()
    Config().evaluation = ConfigNode.from_object({"type": "lighteval"})

    evaluator = evaluator_registry.get()

    assert isinstance(evaluator, LightevalEvaluator)
    _clear_evaluation_config()


def test_lighteval_fast_preset_contains_expected_tasks(temp_config):
    from plato.evaluators.lighteval import _resolve_preset

    preset = _resolve_preset("smollm_round_fast")

    assert preset["tasks"] == ["ifeval", "hellaswag", "arc_easy", "arc_challenge", "piqa"]
    assert preset["primary_metric"] == "ifeval_avg"


def test_lighteval_evaluator_normalizes_metrics(monkeypatch, temp_config):
    from plato.evaluators.lighteval import LightevalEvaluator

    monkeypatch.setattr(
        "plato.evaluators.lighteval._resolve_model_reference",
        lambda request: "/tmp/mock-model",
    )
    monkeypatch.setattr(
        "plato.evaluators.lighteval._run_lighteval_pipeline",
        lambda **kwargs: {
            "ifeval": 0.31,
            "hellaswag": 0.44,
            "arc_easy": 0.35,
            "arc_challenge": 0.25,
            "piqa": 0.61,
        },
    )

    result = LightevalEvaluator(
        {"type": "lighteval", "preset": "smollm_round_fast"}
    ).evaluate(EvaluationInput(model=object(), tokenizer=object()))

    assert result.evaluator == "lighteval"
    assert result.primary_metric == "ifeval_avg"
    assert result.metrics == {
        "ifeval_avg": 0.31,
        "hellaswag": 0.44,
        "arc_easy": 0.35,
        "arc_challenge": 0.25,
        "arc_avg": 0.30,
        "piqa": 0.61,
    }
    assert result.higher_is_better["ifeval_avg"] is True
    assert result.metadata["tasks"] == [
        "ifeval",
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "piqa",
    ]


def test_lighteval_evaluator_raises_helpful_import_error(monkeypatch, temp_config):
    from plato.evaluators.lighteval import LightevalEvaluator

    monkeypatch.setattr(
        "plato.evaluators.lighteval._resolve_model_reference",
        lambda request: "/tmp/mock-model",
    )

    def _raise_import_error(**kwargs):
        raise ImportError("No module named lighteval")

    monkeypatch.setattr(
        "plato.evaluators.lighteval._run_lighteval_pipeline",
        _raise_import_error,
    )

    with pytest.raises(ImportError, match="optional dependency"):
        LightevalEvaluator(
            {"type": "lighteval", "preset": "smollm_round_fast"}
        ).evaluate(EvaluationInput(model=object(), tokenizer=object()))
