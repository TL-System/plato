"""Helpers for logging structured evaluator outputs on the server side."""

from __future__ import annotations

import re
from numbers import Real
from typing import Any

from plato.evaluators.runner import EVALUATION_PRIMARY_KEY, EVALUATION_RESULTS_KEY

LIGHTEVAL_TASK_COLUMN_ALIASES = {
    "arc:easy": "arc_easy",
    "arc:challenge": "arc_challenge",
    "arc:_average": "arc_avg",
    "piqa_hf": "piqa",
}


def _state_from_trainer(trainer: Any | None) -> dict[str, Any]:
    """Return the mutable context state stored on a trainer, if any."""
    context = getattr(trainer, "context", None)
    state = getattr(context, "state", None)
    return state if isinstance(state, dict) else {}


def _is_numeric_metric(value: Any) -> bool:
    """Return whether a value is a numeric scalar worth flattening."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _sanitize_metric_key(value: str) -> str:
    """Convert a nested metric path component into a stable CSV column suffix."""
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    return re.sub(r"_+", "_", sanitized)


def _canonical_lighteval_task_name(task_name: str) -> str:
    """Normalize Lighteval task ids into stable CSV-friendly names."""
    prefix, separator, suffix = task_name.rpartition(":")
    if separator and suffix.isdigit():
        task_name = prefix

    aliased_name = LIGHTEVAL_TASK_COLUMN_ALIASES.get(task_name, task_name)
    return _sanitize_metric_key(aliased_name)


def _extract_lighteval_logged_items(payload: dict[str, Any]) -> dict[str, float]:
    """Flatten detailed numeric Lighteval task metrics for CSV logging."""
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return {}

    raw_metrics = metadata.get("raw_metrics")
    if not isinstance(raw_metrics, dict):
        return {}

    logged_items: dict[str, float] = {}
    for task_name, task_metrics in raw_metrics.items():
        if not isinstance(task_metrics, dict):
            continue

        task_key = _canonical_lighteval_task_name(str(task_name))
        for metric_name, metric_value in task_metrics.items():
            if _is_numeric_metric(metric_value):
                column_name = (
                    f"evaluation_{task_key}_{_sanitize_metric_key(str(metric_name))}"
                )
                logged_items[column_name] = float(metric_value)

    return logged_items


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

        if payload.get("evaluator") == "lighteval":
            logged_items.update(_extract_lighteval_logged_items(payload))

    return logged_items
