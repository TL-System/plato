from __future__ import annotations

import sys
import types
from enum import Enum, auto
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


def test_lighteval_pipeline_matches_supported_api_contract(monkeypatch, temp_config):
    from plato.evaluators.lighteval import (
        LightevalModelReference,
        _run_lighteval_pipeline,
    )

    calls = {}

    class FakeParallelismManager(Enum):
        ACCELERATE = auto()

    class FakePipelineParameters:
        def __init__(self, launcher_type):
            calls["launcher_type"] = launcher_type

    class FakeEvaluationTracker:
        def __init__(self, output_dir, save_details=False):
            assert output_dir
            calls["tracker_output_dir"] = output_dir
            calls["save_details"] = save_details

    class FakeTransformersModelConfig:
        def __init__(self, model_name, tokenizer=None):
            calls["model_name"] = model_name
            calls["tokenizer"] = tokenizer

    class FakePipeline:
        def __init__(
            self,
            tasks,
            pipeline_parameters,
            evaluation_tracker,
            model_config,
        ):
            calls["tasks"] = tasks
            calls["pipeline_parameters"] = pipeline_parameters
            calls["evaluation_tracker"] = evaluation_tracker
            calls["model_config"] = model_config

        def evaluate(self):
            calls["evaluated"] = True

        def get_results(self):
            calls["get_results"] = True
            return {
                "results": {
                    "ifeval": {
                        "prompt_level_strict_acc": 0.30,
                        "prompt_level_loose_acc": 0.50,
                    },
                    "hellaswag": {"exact_match": 0.44},
                    "arc:easy": {"loglikelihood_acc": 0.35},
                    "arc:challenge": {"loglikelihood_acc": 0.25},
                    "piqa": {"exact_match": 0.61},
                }
            }

    lighteval_pkg = types.ModuleType("lighteval")
    logging_pkg = types.ModuleType("lighteval.logging")
    tracker_pkg = types.ModuleType("lighteval.logging.evaluation_tracker")
    tracker_pkg.EvaluationTracker = FakeEvaluationTracker
    models_pkg = types.ModuleType("lighteval.models")
    transformers_pkg = types.ModuleType("lighteval.models.transformers")
    transformers_model_pkg = types.ModuleType(
        "lighteval.models.transformers.transformers_model"
    )
    transformers_model_pkg.TransformersModelConfig = FakeTransformersModelConfig
    pipeline_pkg = types.ModuleType("lighteval.pipeline")
    pipeline_pkg.Pipeline = FakePipeline
    pipeline_pkg.PipelineParameters = FakePipelineParameters
    pipeline_pkg.ParallelismManager = FakeParallelismManager

    monkeypatch.setitem(sys.modules, "lighteval", lighteval_pkg)
    monkeypatch.setitem(sys.modules, "lighteval.logging", logging_pkg)
    monkeypatch.setitem(
        sys.modules,
        "lighteval.logging.evaluation_tracker",
        tracker_pkg,
    )
    monkeypatch.setitem(sys.modules, "lighteval.models", models_pkg)
    monkeypatch.setitem(
        sys.modules,
        "lighteval.models.transformers",
        transformers_pkg,
    )
    monkeypatch.setitem(
        sys.modules,
        "lighteval.models.transformers.transformers_model",
        transformers_model_pkg,
    )
    monkeypatch.setitem(sys.modules, "lighteval.pipeline", pipeline_pkg)

    results = _run_lighteval_pipeline(
        model_reference=LightevalModelReference(
            model_name="/tmp/mock-model",
            tokenizer_name="/tmp/mock-tokenizer",
        ),
        tasks=["ifeval", "hellaswag", "arc_easy", "arc_challenge", "piqa"],
        backend="transformers",
        config={},
    )

    assert calls["launcher_type"] is FakeParallelismManager.ACCELERATE
    assert calls["model_name"] == "/tmp/mock-model"
    assert calls["tokenizer"] == "/tmp/mock-tokenizer"
    assert calls["save_details"] is False
    assert calls["tasks"] == "ifeval,hellaswag,arc_easy,arc_challenge,piqa"
    assert calls["evaluated"] is True
    assert calls["get_results"] is True
    assert results == {
        "ifeval": {
            "prompt_level_strict_acc": 0.30,
            "prompt_level_loose_acc": 0.50,
        },
        "hellaswag": {"exact_match": 0.44},
        "arc:easy": {"loglikelihood_acc": 0.35},
        "arc:challenge": {"loglikelihood_acc": 0.25},
        "piqa": {"exact_match": 0.61},
    }


def test_lighteval_evaluator_normalizes_metrics(monkeypatch, temp_config):
    from plato.evaluators.lighteval import (
        LightevalEvaluator,
        LightevalModelReference,
    )

    monkeypatch.setattr(
        "plato.evaluators.lighteval._resolve_model_reference",
        lambda request, export_dir=None: LightevalModelReference(
            model_name="/tmp/mock-model",
            tokenizer_name="/tmp/mock-model",
        ),
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


def test_lighteval_normalizes_versioned_task_keys(temp_config):
    from plato.evaluators.lighteval import _normalize_metrics

    metrics = _normalize_metrics(
        {
            "ifeval:0": {
                "prompt_level_strict_acc": 0.30,
                "prompt_level_loose_acc": 0.50,
            },
            "hellaswag:0": {"exact_match": 0.44},
            "arc:easy:0": {"loglikelihood_acc": 0.35},
            "arc:challenge:0": {"loglikelihood_acc": 0.25},
            "piqa:0": {"exact_match": 0.61},
        }
    )

    assert metrics == {
        "ifeval_avg": 0.40,
        "hellaswag": 0.44,
        "arc_easy": 0.35,
        "arc_challenge": 0.25,
        "arc_avg": 0.30,
        "piqa": 0.61,
    }


def test_lighteval_evaluator_cleans_up_temp_exports(monkeypatch, temp_config):
    from plato.evaluators.lighteval import LightevalEvaluator

    captured = {}

    class SaveableArtifact:
        def save_pretrained(self, path: str) -> None:
            Path(path, "artifact.bin").write_text("ok", encoding="utf-8")

    def _mock_pipeline(**kwargs):
        reference = kwargs["model_reference"]
        captured["model_name"] = reference.model_name
        assert Path(reference.model_name).is_dir()
        return {
            "ifeval": {
                "prompt_level_strict_acc": 0.30,
                "prompt_level_loose_acc": 0.50,
            },
            "hellaswag": {"exact_match": 0.44},
            "arc:easy": {"loglikelihood_acc": 0.35},
            "arc:challenge": {"loglikelihood_acc": 0.25},
            "piqa": {"exact_match": 0.61},
        }

    monkeypatch.setattr(
        "plato.evaluators.lighteval._run_lighteval_pipeline",
        _mock_pipeline,
    )

    result = LightevalEvaluator(
        {"type": "lighteval", "preset": "smollm_round_fast"}
    ).evaluate(
        EvaluationInput(model=SaveableArtifact(), tokenizer=SaveableArtifact())
    )

    assert result.metrics["ifeval_avg"] == pytest.approx(0.40)
    assert captured["model_name"]
    assert not Path(captured["model_name"]).exists()


def test_lighteval_evaluator_raises_helpful_import_error(monkeypatch, temp_config):
    from plato.evaluators.lighteval import (
        LightevalEvaluator,
        LightevalModelReference,
    )

    monkeypatch.setattr(
        "plato.evaluators.lighteval._resolve_model_reference",
        lambda request, export_dir=None: LightevalModelReference(
            model_name="/tmp/mock-model",
            tokenizer_name="/tmp/mock-model",
        ),
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
