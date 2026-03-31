"""Lighteval adapter for structured LLM evaluation."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterator

from plato.config import Config
from plato.evaluators import registry
from plato.evaluators.base import EvaluationInput, EvaluationResult, Evaluator

LIGHTEVAL_EVALUATOR = "lighteval"
CUSTOM_TASKS_MODULE = "plato.evaluators.lighteval_tasks"
LIGHTEVAL_PRESETS: dict[str, dict[str, Any]] = {
    "smollm_round_fast": {
        "tasks": ["ifeval", "hellaswag", "arc_easy", "arc_challenge", "piqa"],
        "primary_metric": "ifeval_avg",
    }
}
TASK_ALIASES: dict[str, tuple[str, ...]] = {
    "ifeval": ("ifeval",),
    "hellaswag": ("hellaswag",),
    "arc_easy": ("arc_easy", "arc:easy"),
    "arc_challenge": ("arc_challenge", "arc:challenge"),
    "piqa": ("piqa", "piqa_hf"),
}
TASK_PIPELINE_NAMES: dict[str, str] = {
    "arc_easy": "arc:easy",
    "arc_challenge": "arc:challenge",
    "piqa": "piqa_hf",
}
TASK_METRIC_PREFERENCES: dict[str, tuple[str, ...]] = {
    "hellaswag": ("exact_match", "loglikelihood_acc", "accuracy", "acc"),
    "arc_easy": ("loglikelihood_acc", "exact_match", "accuracy", "acc"),
    "arc_challenge": ("loglikelihood_acc", "exact_match", "accuracy", "acc"),
    "piqa": ("exact_match", "loglikelihood_acc", "accuracy", "acc"),
}


@dataclass(slots=True)
class LightevalModelReference:
    """Resolved model/tokenizer reference for a Lighteval run."""

    model_name: str
    tokenizer_name: str | None = None


def _config_value(config: dict[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_preset(name: str) -> dict[str, Any]:
    if name not in LIGHTEVAL_PRESETS:
        raise ValueError(f"Unknown Lighteval preset: {name}")
    return dict(LIGHTEVAL_PRESETS[name])


def _is_numeric_metric(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _canonical_task_name(task_name: str) -> str:
    prefix, separator, suffix = task_name.rpartition(":")
    if separator and suffix.isdigit():
        return prefix
    return task_name


def _find_task_metrics(
    raw_metrics: dict[str, Any],
    task_name: str,
) -> dict[str, Any] | None:
    aliases = TASK_ALIASES.get(task_name, (task_name,))
    for candidate in aliases:
        task_metrics = raw_metrics.get(candidate)
        if isinstance(task_metrics, dict):
            return task_metrics

    canonical_candidates = {_canonical_task_name(candidate) for candidate in aliases}
    for raw_task_name, task_metrics in raw_metrics.items():
        if (
            isinstance(task_metrics, dict)
            and _canonical_task_name(raw_task_name) in canonical_candidates
        ):
            return task_metrics
    return None


def _preferred_task_value(task_metrics: dict[str, Any], task_name: str) -> float | None:
    for metric_name in TASK_METRIC_PREFERENCES.get(task_name, ()):  # pragma: no branch
        metric_value = task_metrics.get(metric_name)
        if _is_numeric_metric(metric_value):
            return float(metric_value)

    for metric_name, metric_value in task_metrics.items():
        if metric_name.endswith("_stderr"):
            continue
        if _is_numeric_metric(metric_value):
            return float(metric_value)
    return None


def _resolve_pipeline_tasks(tasks: list[str]) -> list[str]:
    """Convert friendly task aliases into concrete Lighteval task ids."""
    return [TASK_PIPELINE_NAMES.get(task, task) for task in tasks]


def _normalize_nested_metrics(raw_metrics: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    ifeval_metrics = _find_task_metrics(raw_metrics, "ifeval")
    if ifeval_metrics is not None:
        strict_acc = ifeval_metrics.get("prompt_level_strict_acc")
        loose_acc = ifeval_metrics.get("prompt_level_loose_acc")
        if _is_numeric_metric(strict_acc) and _is_numeric_metric(loose_acc):
            metrics["ifeval_avg"] = (float(strict_acc) + float(loose_acc)) / 2.0
        elif _is_numeric_metric(strict_acc):
            metrics["ifeval_avg"] = float(strict_acc)
        elif _is_numeric_metric(loose_acc):
            metrics["ifeval_avg"] = float(loose_acc)
        else:
            fallback = _preferred_task_value(ifeval_metrics, "ifeval")
            if fallback is not None:
                metrics["ifeval_avg"] = fallback

    for task_name in ("hellaswag", "arc_easy", "arc_challenge", "piqa"):
        task_metrics = _find_task_metrics(raw_metrics, task_name)
        if task_metrics is None:
            continue
        value = _preferred_task_value(task_metrics, task_name)
        if value is not None:
            metrics[task_name] = value

    if "arc_easy" in metrics and "arc_challenge" in metrics:
        metrics["arc_avg"] = (metrics["arc_easy"] + metrics["arc_challenge"]) / 2.0

    return metrics


def _normalize_metrics(raw_metrics: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    nested_results = raw_metrics.get("results")
    if isinstance(nested_results, dict):
        raw_metrics = nested_results

    if any(isinstance(value, dict) for value in raw_metrics.values()):
        metrics.update(_normalize_nested_metrics(raw_metrics))
        if metrics:
            return metrics

    if "ifeval_avg" in raw_metrics:
        metrics["ifeval_avg"] = float(raw_metrics["ifeval_avg"])
    elif "ifeval" in raw_metrics:
        metrics["ifeval_avg"] = float(raw_metrics["ifeval"])

    for key in ("hellaswag", "arc_easy", "arc_challenge", "piqa"):
        if key in raw_metrics:
            metrics[key] = float(raw_metrics[key])

    if "arc_avg" in raw_metrics:
        metrics["arc_avg"] = float(raw_metrics["arc_avg"])
    elif "arc_easy" in metrics and "arc_challenge" in metrics:
        metrics["arc_avg"] = (metrics["arc_easy"] + metrics["arc_challenge"]) / 2.0

    return metrics


def _resolve_model_reference(
    request: EvaluationInput,
    export_dir: str | None = None,
) -> LightevalModelReference:
    model = request.model
    tokenizer = request.tokenizer
    if export_dir is not None and model is not None and tokenizer is not None:
        save_model = getattr(model, "save_pretrained", None)
        save_tokenizer = getattr(tokenizer, "save_pretrained", None)
        if callable(save_model) and callable(save_tokenizer):
            save_model(export_dir)
            save_tokenizer(export_dir)
            return LightevalModelReference(
                model_name=export_dir,
                tokenizer_name=export_dir,
            )

    trainer_cfg = getattr(Config(), "trainer", None)
    configured_model_name = _config_value(trainer_cfg, "model_name")
    if isinstance(configured_model_name, str) and configured_model_name:
        tokenizer_name = _config_value(
            trainer_cfg,
            "tokenizer_name",
            configured_model_name,
        )
        if not isinstance(tokenizer_name, str) or not tokenizer_name:
            tokenizer_name = configured_model_name
        return LightevalModelReference(
            model_name=configured_model_name,
            tokenizer_name=tokenizer_name,
        )

    raise ValueError(
        "Lighteval evaluator requires either a configured model reference or a model and tokenizer with save_pretrained()."
    )


@contextmanager
def _materialize_model_reference(
    request: EvaluationInput,
) -> Iterator[LightevalModelReference]:
    model = request.model
    tokenizer = request.tokenizer
    save_model = getattr(model, "save_pretrained", None)
    save_tokenizer = getattr(tokenizer, "save_pretrained", None)

    if callable(save_model) and callable(save_tokenizer):
        with tempfile.TemporaryDirectory(prefix="plato-lighteval-") as export_dir:
            yield _resolve_model_reference(request, export_dir=export_dir)
        return

    yield _resolve_model_reference(request)


def _resolve_launcher_type(backend: str, parallelism_manager: Any) -> Any:
    normalized_backend = backend.strip().lower()
    if normalized_backend in {"transformers", "accelerate"}:
        return parallelism_manager.ACCELERATE

    enum_name = normalized_backend.upper()
    if hasattr(parallelism_manager, enum_name):
        return getattr(parallelism_manager, enum_name)

    raise NotImplementedError(
        f"Unsupported Lighteval backend '{backend}'. Supported values include 'transformers'/'accelerate' and available ParallelismManager enum names."
    )


def _run_lighteval_pipeline(
    *,
    model_reference: LightevalModelReference,
    tasks: list[str],
    backend: str,
    config: dict[str, Any] | Any,
) -> dict[str, Any]:
    try:
        from lighteval.logging.evaluation_tracker import EvaluationTracker
        from lighteval.models.transformers.transformers_model import (
            TransformersModelConfig,
        )
        from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Lighteval is an optional dependency. Install it via the project's llm_eval extra to use evaluation.type = 'lighteval'."
        ) from exc

    launcher_type = _resolve_launcher_type(backend, ParallelismManager)
    resolved_tasks = _resolve_pipeline_tasks(tasks)

    with tempfile.TemporaryDirectory(prefix="plato-lighteval-output-") as output_dir:
        tracker = EvaluationTracker(output_dir=output_dir, save_details=False)
        pipeline_parameters = PipelineParameters(
            launcher_type=launcher_type,
            custom_tasks_directory=CUSTOM_TASKS_MODULE,
        )
        model_config = TransformersModelConfig(
            model_name=model_reference.model_name,
            tokenizer=model_reference.tokenizer_name,
        )
        pipeline = Pipeline(
            tasks=",".join(resolved_tasks),
            pipeline_parameters=pipeline_parameters,
            evaluation_tracker=tracker,
            model_config=model_config,
        )
        pipeline.evaluate()

        results = pipeline.get_results()
        if isinstance(results, dict):
            nested_results = results.get("results")
            if isinstance(nested_results, dict):
                return dict(nested_results)

        tracker_results = getattr(tracker, "results", None)
        if isinstance(tracker_results, dict):
            nested_results = tracker_results.get("results")
            if isinstance(nested_results, dict):
                return dict(nested_results)

    raise RuntimeError(
        "Lighteval pipeline did not expose a dictionary of task metrics."
    )


class LightevalEvaluator(Evaluator):
    """Structured evaluator adapter for Lighteval benchmark runs."""

    def evaluate(self, request: EvaluationInput) -> EvaluationResult:
        preset_name = _config_value(self.config, "preset", "smollm_round_fast")
        preset = _resolve_preset(str(preset_name))
        tasks = list(preset["tasks"])
        backend = str(_config_value(self.config, "backend", "transformers"))
        primary_metric = str(
            _config_value(self.config, "primary_metric", preset["primary_metric"])
        )

        try:
            with _materialize_model_reference(request) as model_reference:
                raw_metrics = _run_lighteval_pipeline(
                    model_reference=model_reference,
                    tasks=tasks,
                    backend=backend,
                    config=self.config,
                )
        except ImportError as exc:
            raise ImportError(
                "Lighteval is an optional dependency; install the project's optional dependency to use evaluation.type = 'lighteval'."
            ) from exc

        metrics = _normalize_metrics(raw_metrics)
        if primary_metric not in metrics:
            raise ValueError(
                f"Primary metric '{primary_metric}' missing from normalized Lighteval metrics {sorted(metrics)}."
            )

        return EvaluationResult(
            evaluator=LIGHTEVAL_EVALUATOR,
            primary_metric=primary_metric,
            metrics=metrics,
            higher_is_better={key: True for key in metrics},
            metadata={
                "preset": preset_name,
                "tasks": tasks,
                "backend": backend,
                "raw_metrics": dict(raw_metrics),
            },
        )


registry.register(LIGHTEVAL_EVALUATOR, LightevalEvaluator)
