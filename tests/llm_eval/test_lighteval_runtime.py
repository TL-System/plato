"""Actual Lighteval Pipeline qualification; requires the llm_eval profile."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
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


@pytest.fixture
def readonly_source_permissions(tmp_path, record_property):
    """Restore permissions on synthetic sources after all test assertions."""
    original_modes = {}

    def make_readonly(*roots: Path) -> None:
        for root in dict.fromkeys(roots):
            assert root.resolve().is_relative_to(tmp_path.resolve())
            paths = [*root.rglob("*"), root] if root.is_dir() else [root]
            for path in paths:
                if not path.is_symlink():
                    original_modes.setdefault(
                        path, stat.S_IMODE(path.stat().st_mode)
                    )
                    path.chmod(0o555 if path.is_dir() else 0o444)

    try:
        yield make_readonly
    finally:
        restored = 0
        for path, mode in original_modes.items():
            if path.exists() and not path.is_symlink():
                path.chmod(mode)
                restored += 1
        record_property("readonly_fixture_permissions_restored", restored)


@pytest.mark.parametrize("input_source", ["current-model", "configured-local"])
def test_real_preset_pipeline_normalizes_metrics_and_refreshes_exports(
    runtime_assets, monkeypatch, record_property, tmp_path, input_source
):
    from datasets import load_dataset
    from lighteval.pipeline import Pipeline

    model, tokenizer = runtime_assets
    model_directory = tmp_path / "configured-model"
    if input_source == "configured-local":
        model.save_pretrained(model_directory)
        tokenizer.save_pretrained(model_directory)
        Config().trainer.model_name = str(model_directory)
        Config().trainer.tokenizer_name = str(model_directory)
    request_model = model if input_source == "current-model" else object()
    request_tokenizer = tokenizer if input_source == "current-model" else None
    context = TrainingContext()
    observed_runs = []
    copied_artifacts = []
    original_copytree = shutil.copytree

    def record_copytree(source, destination, *args, **kwargs):
        if Path(source).resolve() == model_directory.resolve():
            copied_artifacts.append(
                sum(
                    path.stat().st_size
                    for path in Path(source).rglob("*")
                    if path.is_file()
                )
            )
        return original_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", record_copytree)
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
    source_before = (
        helpers.snapshot_directory(model_directory)
        if input_source == "configured-local"
        else None
    )
    result = run_configured_evaluation(
        model=request_model, tokenizer=request_tokenizer, context=context
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
        "ifeval:0",
        "hellaswag:0",
        "arc:easy:0",
        "arc:challenge:0",
        "piqa_hf:0",
        "arc:_average:0",
        "all",
    }
    assert context.state[EVALUATION_PRIMARY_KEY]["value"] == pytest.approx(2 / 3)
    assert context.state[EVALUATION_RESULTS_KEY]["lighteval"] == result.to_dict()
    if source_before is not None:
        assert helpers.snapshot_directory(model_directory) == source_before
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)

    helpers.prefer_token(model, 1)
    if input_source == "configured-local":
        model.save_pretrained(model_directory)
        source_before = helpers.snapshot_directory(model_directory)
    second = run_configured_evaluation(
        model=request_model, tokenizer=request_tokenizer, context=context
    )
    assert second is not None
    assert second.metrics == {key: pytest.approx(1 / 3) for key in expected}
    if input_source == "current-model":
        assert observed_runs[0]["export"] != observed_runs[1]["export"]
    else:
        assert observed_runs[0]["cache"] != observed_runs[1]["cache"]
        assert not list(model_directory.rglob("*.parquet"))
        assert helpers.snapshot_directory(model_directory) == source_before
    assert len(copied_artifacts) == (2 if input_source == "configured-local" else 0)
    for observed in observed_runs:
        assert len(observed["responses"]) == 5
        assert all(len(rows) == 3 for rows in observed["responses"].values())
        assert not observed["export"].exists()
        if input_source == "configured-local":
            assert observed["export"] != model_directory
            assert model_directory.is_dir()
        assert not observed["output"].exists()
        assert not observed["cache"].exists()
    for observed, letter, token_id in zip(observed_runs, ("A", "B"), (0, 1)):
        assert all(
            row["sample"]["text"] == [letter]
            for row in observed["responses"]["piqa_hf|0"]
        )
        for row in observed["responses"]["arc:easy|0"]:
            logprobs = row["sample"]["logprobs"]
            assert logprobs[token_id] > logprobs[1 - token_id] + 7.9
    record_property(
        "actual_runtime_rounds",
        json.dumps(
            {
                "normalized": [result.to_dict(), second.to_dict()],
                "cache_responses": [
                    observed["responses"] for observed in observed_runs
                ],
                "input_source": input_source,
                "temporary_artifact_cleanup": True,
                "independent_copy_bytes_per_run": copied_artifacts,
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
    runtime_assets, monkeypatch, fail_on_error, record_property, caplog
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

        with pytest.raises(ValidationError, match="max_length") as error:
            run_configured_evaluation(model=model, tokenizer=tokenizer, context=context)
        error_text = str(error.value)
    else:
        assert (
            run_configured_evaluation(model=model, tokenizer=tokenizer, context=context)
            is None
        )
        assert "continuing without structured evaluation" in caplog.text
        assert "max_length" in caplog.text
        error_text = caplog.text
    assert EVALUATION_RESULTS_KEY not in context.state
    assert EVALUATION_PRIMARY_KEY not in context.state
    assert torch.is_grad_enabled() == original_grad
    assert len(exports) == 1
    assert not exports[0].exists()
    record_property(
        "actual_negative_case",
        json.dumps(
            {
                "fail_on_error": fail_on_error,
                "error": error_text,
                "stale_context_cleared": True,
                "export_removed": True,
                "grad_mode_restored": True,
            }
        ),
    )


def test_real_missing_primary_metric_is_rejected(runtime_assets):
    model, tokenizer = runtime_assets
    Config().evaluation.primary_metric = "missing"
    Config().evaluation.max_samples = 1
    context = TrainingContext()
    context.state[EVALUATION_PRIMARY_KEY] = {"stale": 1}

    with pytest.raises(ValueError, match="Primary metric 'missing' missing"):
        run_configured_evaluation(model=model, tokenizer=tokenizer, context=context)

    assert EVALUATION_PRIMARY_KEY not in context.state
    assert EVALUATION_RESULTS_KEY not in context.state


@pytest.mark.parametrize("tokenizer_source", ["same", "separate"])
def test_real_readonly_local_source_ignores_old_responses(
    runtime_assets,
    tmp_path,
    monkeypatch,
    record_property,
    tokenizer_source,
    readonly_source_permissions,
):
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.transformers.transformers_model import TransformersModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    from plato.evaluators.lighteval import CUSTOM_TASKS_MODULE

    model, tokenizer = runtime_assets
    source = tmp_path / "persistent-model"
    tokenizer_directory = (
        source if tokenizer_source == "same" else tmp_path / "persistent-tokenizer"
    )
    model.save_pretrained(source)
    tokenizer.save_pretrained(tokenizer_directory)
    # Seed genuine legacy cached A responses in the caller-owned source. These
    # remain there; subsequent Plato evaluation must use the new B weights.
    with torch.no_grad():
        seed = Pipeline(
            tasks="piqa_hf",
            pipeline_parameters=PipelineParameters(
                launcher_type=ParallelismManager.ACCELERATE,
                custom_tasks_directory=CUSTOM_TASKS_MODULE,
            ),
            evaluation_tracker=EvaluationTracker(
                output_dir=str(tmp_path / "seed-output")
            ),
            model_config=TransformersModelConfig(
                model_name=str(source),
                tokenizer=str(tokenizer_directory),
                batch_size=1,
                max_length=64,
                dtype="float32",
                device="cpu",
                model_parallel=False,
            ),
        )
        seeded_responses = seed.model.greedy_until(seed.documents_dict["piqa_hf|0"])
        assert [response.text for response in seeded_responses] == [["A"]] * 3
        seed.model.cleanup()
    assert len(list(source.rglob("*.parquet"))) == 1
    helpers.prefer_token(model, 1)
    model.save_pretrained(source)

    # Model files in a local Hugging Face snapshot can be file links to blobs.
    blob = tmp_path / "model-blob.safetensors"
    (source / "model.safetensors").rename(blob)
    (source / "model.safetensors").symlink_to(blob)
    sentinel = source / "nested" / "preserved.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("old source artifacts remain unchanged")
    readonly_source_permissions(source, tokenizer_directory, blob)
    source_before = helpers.snapshot_directory(source)
    tokenizer_before = helpers.snapshot_directory(tokenizer_directory)
    blob_before = blob.read_bytes()
    permissions_enforced = os.geteuid() != 0
    if permissions_enforced:
        with pytest.raises(PermissionError):
            (source / "forbidden-write").write_text("must fail")
    Config().trainer.model_name = str(source)
    Config().trainer.tokenizer_name = str(tokenizer_directory)
    observed = {}
    original_evaluate = Pipeline.evaluate

    def record_real_evaluate(pipeline):
        original_evaluate(pipeline)
        copied_model = Path(pipeline.model_config.model_name)
        copied_tokenizer = Path(pipeline.model_config.tokenizer)
        assert not (copied_model / "model.safetensors").is_symlink()
        assert not (copied_model / "model.safetensors").samefile(blob)
        assert copied_model.stat().st_mode & stat.S_IWUSR
        observed.update(
            model=copied_model,
            tokenizer=copied_tokenizer,
            output=Path(pipeline.evaluation_tracker.output_dir),
            cache=pipeline.model._cache.cache_dir,
        )

    monkeypatch.setattr(Pipeline, "evaluate", record_real_evaluate)
    result = run_configured_evaluation(model=object(), context=TrainingContext())

    assert result is not None
    assert all(value == pytest.approx(1 / 3) for value in result.metrics.values())
    assert observed["model"] != source
    assert observed["tokenizer"] != tokenizer_directory
    assert (observed["model"] == observed["tokenizer"]) == (tokenizer_source == "same")
    assert all(not path.exists() for path in observed.values())
    assert helpers.snapshot_directory(source) == source_before
    assert helpers.snapshot_directory(tokenizer_directory) == tokenizer_before
    assert blob.read_bytes() == blob_before
    assert Config().trainer.model_name == str(source)
    assert Config().trainer.tokenizer_name == str(tokenizer_directory)
    record_property(
        "actual_readonly_source",
        json.dumps(
            {
                "tokenizer_source": tokenizer_source,
                "seeded_response_text": [
                    response.text for response in seeded_responses
                ],
                "result": result.to_dict(),
                "source_tree": source_before,
                "source_unchanged": True,
                "readonly_permission_enforced": permissions_enforced,
                "owned_copies_and_cache_removed": True,
            }
        ),
    )


@pytest.mark.parametrize("fail_on_error", [False, True])
def test_failure_after_actual_inference_cleans_local_copies(
    runtime_assets, tmp_path, monkeypatch, record_property, fail_on_error
):
    from lighteval.pipeline import Pipeline

    model, tokenizer = runtime_assets
    source = tmp_path / "model"
    model.save_pretrained(source)
    tokenizer.save_pretrained(source)
    Config().trainer.model_name = str(source)
    Config().trainer.tokenizer_name = str(source)
    Config().evaluation.max_samples = 1
    Config().evaluation.fail_on_error = fail_on_error
    source_before = helpers.snapshot_directory(source)
    failure = RuntimeError("Deliberate error after actual inference and cache writes")
    observed = {}
    original_run_model = Pipeline._run_model

    def fail_after_real_work(pipeline):
        outputs = original_run_model(pipeline)
        assert sum(len(responses) for responses in outputs.values()) == 5
        cache = pipeline.model._cache.cache_dir
        assert len(list(cache.rglob("*.parquet"))) == 5
        observed.update(
            model=Path(pipeline.model_config.model_name),
            output=Path(pipeline.evaluation_tracker.output_dir),
            cache=cache,
        )
        raise failure

    monkeypatch.setattr(Pipeline, "_run_model", fail_after_real_work)
    context = TrainingContext()
    context.state.update(
        {EVALUATION_RESULTS_KEY: {"stale": {}}, EVALUATION_PRIMARY_KEY: {"stale": 1}}
    )
    if fail_on_error:
        with pytest.raises(RuntimeError) as error:
            run_configured_evaluation(model=object(), context=context)
        assert error.value is failure
    else:
        assert run_configured_evaluation(model=object(), context=context) is None

    assert all(not path.exists() for path in observed.values())
    assert helpers.snapshot_directory(source) == source_before
    assert EVALUATION_RESULTS_KEY not in context.state
    assert EVALUATION_PRIMARY_KEY not in context.state
    record_property(
        "actual_post_inference_failure",
        json.dumps(
            {
                "fail_on_error": fail_on_error,
                "response_count_before_injected_failure": 5,
                "cache_files_before_injected_failure": 5,
                "source_unchanged": True,
                "owned_copies_and_cache_removed": True,
                "stale_context_cleared": True,
                "exception": str(failure),
            }
        ),
    )


def test_real_hub_style_response_cache_is_per_run(record_property):
    from lighteval.models.transformers.transformers_model import TransformersModelConfig
    from lighteval.utils.cache_management import SampleCache

    paths = []
    with tempfile.TemporaryDirectory(prefix="plato-lighteval-cache-contract-") as owned:
        for index in (1, 2):
            cache_root = Path(owned) / str(index) / "sample-cache"
            config = TransformersModelConfig(
                model_name="example/model", cache_dir=str(cache_root)
            )
            cache = SampleCache(config)
            assert cache.cache_dir.is_relative_to(cache_root)
            assert cache.existing_indices == {}
            paths.append(cache.cache_dir)
        assert paths[0] != paths[1]
    assert all(not path.exists() for path in paths)
    record_property(
        "actual_hub_style_cache_contract",
        "Validated actual SampleCache path isolation; no Hub model inference claimed.",
    )
