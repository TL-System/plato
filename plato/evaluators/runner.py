"""Runner utilities for configured structured evaluators."""

from __future__ import annotations

from typing import Any

from plato.evaluators.base import EvaluationInput, EvaluationResult
from plato.evaluators import registry
from plato.trainers.strategies.base import TrainingContext

EVALUATION_RESULTS_KEY = "evaluation_results"
EVALUATION_PRIMARY_KEY = "evaluation_primary"


def run_configured_evaluation(
    *,
    model: Any,
    context: TrainingContext,
    trainer: Any | None = None,
    tokenizer: Any | None = None,
    config: dict[str, Any] | None = None,
    testset: Any | None = None,
    sampler: Any | None = None,
    local_metric: float | None = None,
) -> EvaluationResult | None:
    """Run the configured evaluator, storing normalized output in context state."""
    evaluator = registry.get(allow_missing=True)
    if evaluator is None:
        context.state.pop(EVALUATION_RESULTS_KEY, None)
        context.state.pop(EVALUATION_PRIMARY_KEY, None)
        return None

    request = EvaluationInput(
        model=model,
        trainer=trainer,
        tokenizer=tokenizer,
        context=context,
        config=config,
        testset=testset,
        sampler=sampler,
        local_metric=local_metric,
    )
    result = evaluator.evaluate(request)
    payload = result.to_dict()

    all_results = context.state.setdefault(EVALUATION_RESULTS_KEY, {})
    all_results[result.evaluator] = payload
    context.state[EVALUATION_PRIMARY_KEY] = {
        "evaluator": result.evaluator,
        "metric": result.primary_metric,
        "value": result.primary_value,
    }
    return result
