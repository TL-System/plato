"""Lighteval adapter for structured LLM evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from plato.evaluators import registry
from plato.evaluators.base import EvaluationInput, EvaluationResult, Evaluator

LIGHTEVAL_EVALUATOR = "lighteval"
LIGHTEVAL_PRESETS: dict[str, dict[str, Any]] = {
    "smollm_round_fast": {
        "tasks": ["ifeval", "hellaswag", "arc_easy", "arc_challenge", "piqa"],
        "primary_metric": "ifeval_avg",
    }
}


def _config_value(config: dict[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_preset(name: str) -> dict[str, Any]:
    if name not in LIGHTEVAL_PRESETS:
        raise ValueError(f"Unknown Lighteval preset: {name}")
    return dict(LIGHTEVAL_PRESETS[name])


def _normalize_metrics(raw_metrics: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}

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


def _resolve_model_reference(request: EvaluationInput) -> str:
    trainer = request.trainer
    if trainer is not None:
        configured_reference = _config_value(getattr(trainer, "config", None), "model_name")
        if isinstance(configured_reference, str) and configured_reference:
            return configured_reference

    model = request.model
    tokenizer = request.tokenizer
    if model is not None and tokenizer is not None:
        save_model = getattr(model, "save_pretrained", None)
        save_tokenizer = getattr(tokenizer, "save_pretrained", None)
        if callable(save_model) and callable(save_tokenizer):
            export_dir = Path(tempfile.mkdtemp(prefix="plato-lighteval-"))
            save_model(str(export_dir))
            save_tokenizer(str(export_dir))
            return str(export_dir)

    raise ValueError(
        "Lighteval evaluator requires either a trainer/model reference or a model and tokenizer with save_pretrained()."
    )


def _run_lighteval_pipeline(
    *,
    model_reference: str,
    tasks: list[str],
    backend: str,
    config: dict[str, Any] | Any,
) -> dict[str, Any]:
    try:
        from lighteval.logging.evaluation_tracker import EvaluationTracker
        from lighteval.models.transformers.transformers_model import (
            TransformersModelConfig,
        )
        from lighteval.pipeline import Pipeline, PipelineParameters
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Lighteval is an optional dependency. Install it via the project's llm_eval extra to use evaluation.type = 'lighteval'."
        ) from exc

    if backend != "transformers":
        raise NotImplementedError(
            f"Unsupported Lighteval backend '{backend}'. Only 'transformers' is implemented."
        )

    tracker = EvaluationTracker(output_dir=None, save_details=False)
    pipeline_parameters = PipelineParameters(launcher_type=backend)
    model_config = TransformersModelConfig(pretrained=model_reference)
    pipeline = Pipeline(
        tasks=",".join(tasks),
        pipeline_parameters=pipeline_parameters,
        evaluation_tracker=tracker,
        model_config=model_config,
    )
    pipeline.evaluate()

    if hasattr(tracker, "latest_results") and isinstance(tracker.latest_results, dict):
        return dict(tracker.latest_results)
    if hasattr(pipeline, "results") and isinstance(pipeline.results, dict):
        return dict(pipeline.results)
    raise RuntimeError("Lighteval pipeline did not expose a dictionary of metrics.")


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
            raw_metrics = _run_lighteval_pipeline(
                model_reference=_resolve_model_reference(request),
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
