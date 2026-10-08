"""Runner utilities for configured structured evaluators."""

from __future__ import annotations

import logging
from typing import Any

import torch

from plato.config import Config
from plato.evaluators import registry
from plato.evaluators.base import EvaluationInput, EvaluationResult
from plato.trainers.strategies.base import TrainingContext

EVALUATION_RESULTS_KEY = "evaluation_results"
EVALUATION_PRIMARY_KEY = "evaluation_primary"
LEGACY_OPTIONAL_EVALUATORS = {"nanochat_core"}


def _configured_evaluator_type() -> str | None:
    """Return the configured evaluator type when present."""
    evaluation_cfg = getattr(Config(), "evaluation", None)
    if evaluation_cfg is None:
        return None
    if isinstance(evaluation_cfg, dict):
        evaluator_type = evaluation_cfg.get("type")
    else:
        evaluator_type = getattr(evaluation_cfg, "type", None)
    return evaluator_type if isinstance(evaluator_type, str) and evaluator_type else None


def _evaluation_fail_on_error() -> bool:
    """Return whether configured evaluator failures should abort execution."""
    evaluation_cfg = getattr(Config(), "evaluation", None)
    if evaluation_cfg is None:
        return False

    if isinstance(evaluation_cfg, dict):
        fail_on_error = evaluation_cfg.get("fail_on_error", False)
    else:
        fail_on_error = getattr(evaluation_cfg, "fail_on_error", False)

    return bool(fail_on_error)


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
    evaluator_override: Any | None = None,
) -> EvaluationResult | None:
    """Run the configured evaluator, storing normalized output in context state."""
    evaluator_type = _configured_evaluator_type()
    if evaluator_type is None:
        context.state.pop(EVALUATION_RESULTS_KEY, None)
        context.state.pop(EVALUATION_PRIMARY_KEY, None)
        return None

    evaluator = None
    if evaluator_override is not None:
        override_config = getattr(evaluator_override, "config", None)
        override_type = None
        if isinstance(override_config, dict):
            override_type = override_config.get("type")
        else:
            override_type = getattr(override_config, "type", None)
        if override_type == evaluator_type:
            evaluator = evaluator_override

    if evaluator is None:
        evaluator = registry.get(
            allow_missing=evaluator_type in LEGACY_OPTIONAL_EVALUATORS,
        )

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
    previous_grad_enabled = torch.is_grad_enabled()
    try:
        result = evaluator.evaluate(request)
    except Exception:  # pragma: no cover - exercised via unit tests
        context.state.pop(EVALUATION_RESULTS_KEY, None)
        context.state.pop(EVALUATION_PRIMARY_KEY, None)
        if _evaluation_fail_on_error():
            raise

        logging.exception(
            "Configured evaluator '%s' failed; continuing without structured evaluation. "
            "Set evaluation.fail_on_error = true to make such failures fatal.",
            evaluator_type,
        )
        return None
    finally:
        torch.set_grad_enabled(previous_grad_enabled)

    payload = result.to_dict()

    context.state[EVALUATION_RESULTS_KEY] = {result.evaluator: payload}
    context.state[EVALUATION_PRIMARY_KEY] = {
        "evaluator": result.evaluator,
        "metric": result.primary_metric,
        "value": result.primary_value,
    }
    return result
