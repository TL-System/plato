"""Actual Lighteval Pipeline qualification; requires the llm_eval profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from plato.config import Config, ConfigNode
from plato.evaluators import registry
from plato.evaluators.runner import (
    EVALUATION_PRIMARY_KEY,
    EVALUATION_RESULTS_KEY,
    run_configured_evaluation,
)
from plato.trainers.strategies.base import TrainingContext
from tests.llm_eval import helpers


@pytest.fixture
def runtime_assets(monkeypatch, tmp_path, temp_config):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    for original, local in helpers.make_offline_tasks(tmp_path):
        for attribute in (
            "hf_repo",
            "hf_subset",
            "hf_revision",
            "hf_avail_splits",
            "evaluation_splits",
            "few_shots_split",
            "few_shots_select",
            "generation_size",
        ):
            monkeypatch.setattr(original, attribute, getattr(local, attribute))
    Config().evaluation = ConfigNode.from_object(
        {
            "type": "lighteval",
            "device": "cpu",
            "dtype": "float32",
            "max_length": 64,
            "show_progress": False,
            "fail_on_error": True,
        }
    )
    return helpers.make_model_and_tokenizer()


def test_real_preset_pipeline_normalizes_metrics_and_refreshes_exports(
    runtime_assets, monkeypatch, record_property
):
    from datasets import load_dataset
    from lighteval.pipeline import Pipeline

    model, tokenizer = runtime_assets
    context = TrainingContext()
    observed_runs = []
    original_evaluate = Pipeline.evaluate

    def record_real_evaluate(pipeline):
        original_evaluate(pipeline)
        cache = pipeline.model._cache
        observed_runs.append(
            {
                "export": Path(pipeline.model_config.model_name),
                "output": Path(pipeline.evaluation_tracker.output_dir),
                "cache": cache.cache_dir,
                "responses": {
                    path.parent.parent.name: list(
                        load_dataset("parquet", data_files=str(path), split="train")
                    )
                    for path in cache.cache_dir.rglob("*.parquet")
                },
            }
        )

    monkeypatch.setattr(Pipeline, "evaluate", record_real_evaluate)
    assert registry.get().__class__.__name__ == "LightevalEvaluator"
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    result = run_configured_evaluation(
        model=model, tokenizer=tokenizer, context=context
    )

    assert result is not None
    expected = {
        key: pytest.approx(2 / 3)
        for key in (
            "ifeval_avg",
            "hellaswag",
            "arc_easy",
            "arc_challenge",
            "arc_avg",
            "piqa",
        )
    }
    assert result.metrics == expected
    assert result.primary_metric == "ifeval_avg"
    assert result.primary_value == pytest.approx(2 / 3)
    assert set(result.metadata["raw_metrics"]) == {
        "ifeval|0",
        "hellaswag|0",
        "arc:easy|0",
        "arc:challenge|0",
        "piqa_hf|0",
    }
    assert context.state[EVALUATION_PRIMARY_KEY]["value"] == pytest.approx(2 / 3)
    assert context.state[EVALUATION_RESULTS_KEY]["lighteval"] == result.to_dict()
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)

    helpers.prefer_token(model, 1)
    second = run_configured_evaluation(
        model=model, tokenizer=tokenizer, context=context
    )
    assert second is not None
    assert second.metrics == {key: pytest.approx(1 / 3) for key in expected}
    assert observed_runs[0]["export"] != observed_runs[1]["export"]
    for observed in observed_runs:
        assert len(observed["responses"]) == 5
        assert all(len(rows) == 3 for rows in observed["responses"].values())
        assert not observed["export"].exists()
        assert not observed["output"].exists()
        assert not observed["cache"].exists()
    record_property(
        "actual_runtime_rounds",
        json.dumps(
            {
                "normalized": [result.to_dict(), second.to_dict()],
                "cache_responses": [
                    observed["responses"] for observed in observed_runs
                ],
                "export_cache_output_cleanup": True,
            }
        ),
    )


def test_real_piqa_prompt_and_preset_task_contract(temp_config):
    from lighteval.tasks.registry import Registry

    from plato.evaluators.lighteval import (
        CUSTOM_TASKS_MODULE,
        _resolve_pipeline_tasks,
        _resolve_preset,
    )
    from plato.evaluators.lighteval_tasks import piqa_hf_prompt

    tasks = _resolve_pipeline_tasks(_resolve_preset("smollm_round_fast")["tasks"])
    task_registry = Registry(tasks=",".join(tasks), custom_tasks=CUSTOM_TASKS_MODULE)
    configs = {
        task.name: task.dataset_config for task in task_registry.load_tasks().values()
    }
    assert set(configs) == {
        "ifeval",
        "hellaswag",
        "arc:easy",
        "arc:challenge",
        "piqa_hf",
    }
    for name, metric in (
        ("hellaswag", "em"),
        ("arc:easy", "acc"),
        ("arc:challenge", "acc"),
        ("piqa_hf", "em"),
    ):
        assert [value.metric_name for value in configs[name].metrics] == [metric]
    assert set(configs["ifeval"].metrics[0].metric_name) == {
        "prompt_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_strict_acc",
        "inst_level_loose_acc",
    }
    document = piqa_hf_prompt(
        {"goal": "Choose", "sol1": "first", "sol2": "second", "label": "1"}
    )
    assert document.task_name == ""
    assert document.choices == ["A", "B"]
    assert document.gold_index == 1
    assert document.query.endswith("A. first\nB. second\nAnswer: ")


@pytest.mark.parametrize("fail_on_error", [False, True])
def test_real_model_config_failure_clears_stale_results_and_exports(
    runtime_assets, monkeypatch, fail_on_error
):
    from plato.evaluators import lighteval

    model, tokenizer = runtime_assets
    Config().evaluation.fail_on_error = fail_on_error
    Config().evaluation.max_length = 0
    context = TrainingContext()
    context.state.update(
        {EVALUATION_RESULTS_KEY: {"stale": {}}, EVALUATION_PRIMARY_KEY: {"stale": 1}}
    )
    exports = []
    original_resolve = lighteval._resolve_model_reference

    def record_export(request, export_dir=None):
        reference = original_resolve(request, export_dir)
        exports.append(Path(reference.model_name))
        return reference

    monkeypatch.setattr(lighteval, "_resolve_model_reference", record_export)
    original_grad = torch.is_grad_enabled()
    if fail_on_error:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="max_length"):
            run_configured_evaluation(model=model, tokenizer=tokenizer, context=context)
    else:
        assert (
            run_configured_evaluation(model=model, tokenizer=tokenizer, context=context)
            is None
        )
    assert EVALUATION_RESULTS_KEY not in context.state
    assert EVALUATION_PRIMARY_KEY not in context.state
    assert torch.is_grad_enabled() == original_grad
    assert len(exports) == 1
    assert not exports[0].exists()
