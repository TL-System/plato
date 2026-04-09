from __future__ import annotations

import sys
import types
from enum import Enum, auto
from pathlib import Path
from typing import Any, cast

import pytest

from plato.config import Config, ConfigNode
from plato.evaluators.base import EvaluationInput


def _clear_evaluation_config() -> None:
    if hasattr(Config, "evaluation"):
        delattr(Config, "evaluation")


def _fake_module(name: str, **attributes: object) -> Any:
    module = cast(Any, types.ModuleType(name))
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


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


def test_lighteval_pipeline_task_resolution_maps_arc_aliases(temp_config):
    from plato.evaluators.lighteval import _resolve_pipeline_tasks

    assert _resolve_pipeline_tasks(
        ["ifeval", "hellaswag", "arc_easy", "arc_challenge", "piqa"]
    ) == ["ifeval", "hellaswag", "arc:easy", "arc:challenge", "piqa_hf"]


def test_lighteval_pipeline_matches_supported_api_contract(monkeypatch, temp_config):
    from plato.evaluators.lighteval import (
        Config,
        LightevalModelReference,
        _run_lighteval_pipeline,
    )

    calls = {}
    progress_updates: list[str] = []

    class FakeProgress:
        def advance(self, message: str) -> None:
            progress_updates.append(message)

    class FakeParallelismManager(Enum):
        ACCELERATE = auto()

    class FakePipelineParameters:
        def __init__(self, launcher_type, custom_tasks_directory=None, max_samples=None):
            calls["launcher_type"] = launcher_type
            calls["custom_tasks_directory"] = custom_tasks_directory
            calls["max_samples"] = max_samples

    class FakeEvaluationTracker:
        def __init__(self, output_dir, save_details=False):
            assert output_dir
            calls["tracker_output_dir"] = output_dir
            calls["save_details"] = save_details

    class FakeTransformersModelConfig:
        def __init__(
            self,
            model_name,
            tokenizer=None,
            batch_size=None,
            max_length=None,
            model_parallel=None,
            dtype=None,
            device=None,
        ):
            calls["model_name"] = model_name
            calls["tokenizer"] = tokenizer
            calls["batch_size"] = batch_size
            calls["max_length"] = max_length
            calls["model_parallel"] = model_parallel
            calls["dtype"] = dtype
            calls["device"] = device

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

    lighteval_pkg = _fake_module("lighteval")
    logging_pkg = _fake_module("lighteval.logging")
    tracker_pkg = _fake_module(
        "lighteval.logging.evaluation_tracker",
        EvaluationTracker=FakeEvaluationTracker,
    )
    models_pkg = _fake_module("lighteval.models")
    transformers_pkg = _fake_module("lighteval.models.transformers")
    transformers_model_pkg = _fake_module(
        "lighteval.models.transformers.transformers_model",
        TransformersModelConfig=FakeTransformersModelConfig,
    )
    pipeline_pkg = _fake_module(
        "lighteval.pipeline",
        Pipeline=FakePipeline,
        PipelineParameters=FakePipelineParameters,
        ParallelismManager=FakeParallelismManager,
    )

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
    monkeypatch.setattr(Config, "device", staticmethod(lambda: "cuda:0"))

    results = _run_lighteval_pipeline(
        model_reference=LightevalModelReference(
            model_name="/tmp/mock-model",
            tokenizer_name="/tmp/mock-tokenizer",
        ),
        tasks=["ifeval", "hellaswag", "arc_easy", "arc_challenge", "piqa"],
        backend="transformers",
        config={},
        progress=FakeProgress(),
    )

    assert calls["launcher_type"] is FakeParallelismManager.ACCELERATE
    assert calls["model_name"] == "/tmp/mock-model"
    assert calls["tokenizer"] == "/tmp/mock-tokenizer"
    assert calls["batch_size"] == 1
    assert calls["max_length"] is None
    assert calls["model_parallel"] is False
    assert calls["dtype"] is None
    assert calls["device"] == "cuda:0"
    assert calls["save_details"] is False
    assert calls["custom_tasks_directory"] == "plato.evaluators.lighteval_tasks"
    assert calls["max_samples"] is None
    assert calls["tasks"] == "ifeval,hellaswag,arc:easy,arc:challenge,piqa_hf"
    assert calls["evaluated"] is True
    assert calls["get_results"] is True
    assert progress_updates == [
        "Initializing Lighteval for 5 task(s)",
        "Running benchmark tasks",
        "Collecting evaluation metrics",
    ]
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


def test_lighteval_pipeline_forwards_runtime_overrides(monkeypatch, temp_config):
    from plato.evaluators.lighteval import (
        Config,
        LightevalModelReference,
        _run_lighteval_pipeline,
    )

    calls = {}

    class FakeParallelismManager(Enum):
        ACCELERATE = auto()

    class FakePipelineParameters:
        def __init__(self, launcher_type, custom_tasks_directory=None, max_samples=None):
            del launcher_type, custom_tasks_directory, max_samples

    class FakeEvaluationTracker:
        def __init__(self, output_dir, save_details=False):
            del output_dir, save_details

    class FakeTransformersModelConfig:
        def __init__(self, model_name, tokenizer=None, **kwargs):
            del model_name, tokenizer
            calls.update(kwargs)

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def evaluate(self):
            return None

        def get_results(self):
            return {"results": {"ifeval": 0.31}}

    lighteval_pkg = _fake_module("lighteval")
    logging_pkg = _fake_module("lighteval.logging")
    tracker_pkg = _fake_module(
        "lighteval.logging.evaluation_tracker",
        EvaluationTracker=FakeEvaluationTracker,
    )
    models_pkg = _fake_module("lighteval.models")
    transformers_pkg = _fake_module("lighteval.models.transformers")
    transformers_model_pkg = _fake_module(
        "lighteval.models.transformers.transformers_model",
        TransformersModelConfig=FakeTransformersModelConfig,
    )
    pipeline_pkg = _fake_module(
        "lighteval.pipeline",
        Pipeline=FakePipeline,
        PipelineParameters=FakePipelineParameters,
        ParallelismManager=FakeParallelismManager,
    )

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
    monkeypatch.setattr(Config, "device", staticmethod(lambda: "cuda:0"))

    _run_lighteval_pipeline(
        model_reference=LightevalModelReference(
            model_name="/tmp/mock-model",
            tokenizer_name="/tmp/mock-tokenizer",
        ),
        tasks=["ifeval"],
        backend="transformers",
        config={
            "batch_size": 2,
            "max_length": 1024,
            "model_parallel": False,
            "dtype": "bfloat16",
            "device": "cuda:1",
        },
    )

    assert calls == {
        "batch_size": 2,
        "max_length": 1024,
        "model_parallel": False,
        "dtype": "bfloat16",
        "device": "cuda:1",
    }


def test_lighteval_pipeline_uses_trainer_precision_for_default_dtype(
    monkeypatch, temp_config
):
    from plato.evaluators.lighteval import (
        Config,
        LightevalModelReference,
        _run_lighteval_pipeline,
    )

    cfg = Config()
    cfg.trainer.bf16 = True
    if hasattr(cfg.trainer, "fp16"):
        cfg.trainer.fp16 = False

    captured = {}

    class FakeParallelismManager(Enum):
        ACCELERATE = auto()

    class FakePipelineParameters:
        def __init__(self, launcher_type, custom_tasks_directory=None, max_samples=None):
            del launcher_type, custom_tasks_directory, max_samples

    class FakeEvaluationTracker:
        def __init__(self, output_dir, save_details=False):
            del output_dir, save_details

    class FakeTransformersModelConfig:
        def __init__(self, model_name, tokenizer=None, **kwargs):
            del model_name, tokenizer
            captured.update(kwargs)

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def evaluate(self):
            return None

        def get_results(self):
            return {"results": {"ifeval": 0.31}}

    lighteval_pkg = _fake_module("lighteval")
    logging_pkg = _fake_module("lighteval.logging")
    tracker_pkg = _fake_module(
        "lighteval.logging.evaluation_tracker",
        EvaluationTracker=FakeEvaluationTracker,
    )
    models_pkg = _fake_module("lighteval.models")
    transformers_pkg = _fake_module("lighteval.models.transformers")
    transformers_model_pkg = _fake_module(
        "lighteval.models.transformers.transformers_model",
        TransformersModelConfig=FakeTransformersModelConfig,
    )
    pipeline_pkg = _fake_module(
        "lighteval.pipeline",
        Pipeline=FakePipeline,
        PipelineParameters=FakePipelineParameters,
        ParallelismManager=FakeParallelismManager,
    )

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
    monkeypatch.setattr(Config, "device", staticmethod(lambda: "cuda:0"))

    _run_lighteval_pipeline(
        model_reference=LightevalModelReference(
            model_name="/tmp/mock-model",
            tokenizer_name="/tmp/mock-tokenizer",
        ),
        tasks=["ifeval"],
        backend="transformers",
        config={},
    )

    assert captured["dtype"] == "bfloat16"
    assert captured["model_parallel"] is False
    assert captured["device"] == "cuda:0"


def test_lighteval_pipeline_forwards_max_samples_to_each_task(monkeypatch, temp_config):
    from plato.evaluators.lighteval import (
        Config,
        LightevalModelReference,
        _run_lighteval_pipeline,
    )

    captured = {}

    class FakeParallelismManager(Enum):
        ACCELERATE = auto()

    class FakePipelineParameters:
        def __init__(self, launcher_type, custom_tasks_directory=None, max_samples=None):
            del launcher_type, custom_tasks_directory
            captured["max_samples"] = max_samples

    class FakeEvaluationTracker:
        def __init__(self, output_dir, save_details=False):
            del output_dir, save_details

    class FakeTransformersModelConfig:
        def __init__(self, model_name, tokenizer=None, **kwargs):
            del model_name, tokenizer, kwargs

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def evaluate(self):
            return None

        def get_results(self):
            return {"results": {"ifeval": 0.31}}

    lighteval_pkg = _fake_module("lighteval")
    logging_pkg = _fake_module("lighteval.logging")
    tracker_pkg = _fake_module(
        "lighteval.logging.evaluation_tracker",
        EvaluationTracker=FakeEvaluationTracker,
    )
    models_pkg = _fake_module("lighteval.models")
    transformers_pkg = _fake_module("lighteval.models.transformers")
    transformers_model_pkg = _fake_module(
        "lighteval.models.transformers.transformers_model",
        TransformersModelConfig=FakeTransformersModelConfig,
    )
    pipeline_pkg = _fake_module(
        "lighteval.pipeline",
        Pipeline=FakePipeline,
        PipelineParameters=FakePipelineParameters,
        ParallelismManager=FakeParallelismManager,
    )

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
    monkeypatch.setattr(Config, "device", staticmethod(lambda: "cuda:0"))

    _run_lighteval_pipeline(
        model_reference=LightevalModelReference(
            model_name="/tmp/mock-model",
            tokenizer_name="/tmp/mock-tokenizer",
        ),
        tasks=["ifeval"],
        backend="transformers",
        config={"max_samples": 32},
    )

    assert captured["max_samples"] == 32


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
            "piqa_hf:0": {"exact_match": 0.61},
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

    with pytest.raises(
        ImportError,
        match="runtime dependencies.*Original error: No module named lighteval",
    ):
        LightevalEvaluator(
            {"type": "lighteval", "preset": "smollm_round_fast"}
        ).evaluate(EvaluationInput(model=object(), tokenizer=object()))
