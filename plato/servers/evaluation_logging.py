"""Helpers for logging structured evaluator outputs on the server side."""

from __future__ import annotations

import json
import os
from numbers import Real
from typing import Any

from plato.config import Config
from plato.evaluators.runner import EVALUATION_PRIMARY_KEY, EVALUATION_RESULTS_KEY


def _state_from_trainer(trainer: Any | None) -> dict[str, Any]:
    """Return the mutable context state stored on a trainer, if any."""
    context = getattr(trainer, "context", None)
    state = getattr(context, "state", None)
    return state if isinstance(state, dict) else {}


def _is_numeric_metric(value: Any) -> bool:
    """Return whether a value is a numeric scalar worth flattening."""
    return isinstance(value, Real) and not isinstance(value, bool)


def extract_logged_items(trainer: Any | None) -> dict[str, float]:
    """Flatten numeric evaluator summary metrics for server runtime logs."""
    state = _state_from_trainer(trainer)
    logged_items: dict[str, float] = {}

    primary = state.get(EVALUATION_PRIMARY_KEY)
    if isinstance(primary, dict) and _is_numeric_metric(primary.get("value")):
        logged_items["evaluation_primary_value"] = primary["value"]

    results = state.get(EVALUATION_RESULTS_KEY)
    if not isinstance(results, dict):
        return logged_items

    for payload in results.values():
        if not isinstance(payload, dict):
            continue

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue

        for metric_name, metric_value in metrics.items():
            if _is_numeric_metric(metric_value):
                logged_items[f"evaluation_{metric_name}"] = metric_value

    return logged_items


def persist_jsonl(
    *, trainer: Any | None, current_round: int, accuracy: float | int | None
) -> None:
    """Append the full structured evaluator payload to a JSONL sidecar."""
    state = _state_from_trainer(trainer)
    results = state.get(EVALUATION_RESULTS_KEY)
    if not isinstance(results, dict) or not results:
        return

    payload = {
        "round": current_round,
        "accuracy": accuracy,
        "evaluation_primary": state.get(EVALUATION_PRIMARY_KEY),
        "evaluation_results": results,
    }

    result_path = Config().params["result_path"]
    os.makedirs(result_path, exist_ok=True)

    sidecar_path = os.path.join(result_path, f"{os.getpid()}_evaluation.jsonl")
    with open(sidecar_path, "a", encoding="utf-8") as sidecar_file:
        json.dump(payload, sidecar_file, sort_keys=True, default=str)
        sidecar_file.write("\n")
