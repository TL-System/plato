"""
Adapter utilities to run Nanochat's CORE evaluation benchmark within Plato.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import json
import logging
import os
import random
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Nanochat CORE evaluation requires PyTorch. "
        "Install the `nanochat` extra (includes torch)."
    ) from exc

import requests
import yaml

from plato.config import Config
from plato.evaluators.base import EvaluationInput, EvaluationResult, Evaluator
from plato.utils.third_party import ThirdPartyImportError, ensure_nanochat_importable

LOGGER = logging.getLogger(__name__)

# URL for the CORE evaluation bundle
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
NANOCHAT_CORE_EVALUATOR = "nanochat_core"
NANOCHAT_CORE_RESULTS_KEY = "nanochat_core_results"


def _config_value(config: dict[str, Any] | Any, key: str, default: Any = None) -> Any:
    """Read a config value from either a mapping or attribute container."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _context_state(context: Any | None) -> dict[str, Any] | None:
    """Return the evaluator context state mapping when present."""
    state = getattr(context, "state", None)
    return state if isinstance(state, dict) else None


def _core_metadata(results: dict[str, Any]) -> dict[str, Any]:
    """Extract structured CORE task outputs for logging/persistence."""
    metadata: dict[str, Any] = {}

    raw_results = results.get("results")
    if isinstance(raw_results, dict):
        metadata["results"] = dict(raw_results)

    centered_results = results.get("centered_results")
    if isinstance(centered_results, dict):
        metadata["centered_results"] = dict(centered_results)

    return metadata


class NanochatCoreEvaluator(Evaluator):
    """Structured evaluator adapter for Nanochat's CORE benchmark."""

    def _resolve_results(self, request: EvaluationInput) -> dict[str, Any]:
        state = _context_state(request.context)
        cached_results = None if state is None else state.get(NANOCHAT_CORE_RESULTS_KEY)
        if isinstance(cached_results, dict) and "core_metric" in cached_results:
            return cached_results

        max_per_task = _config_value(self.config, "max_per_task", -1)
        max_per_task_value = -1 if max_per_task is None else int(max_per_task)

        return run_core_evaluation(
            request.model,
            tokenizer=request.tokenizer,
            bundle_dir=_config_value(self.config, "bundle_dir", None),
            max_per_task=max_per_task_value,
            device=getattr(request.context, "device", None),
        )

    def evaluate(self, request: EvaluationInput) -> EvaluationResult:
        results = self._resolve_results(request)
        core_metric = float(results["core_metric"])

        state = _context_state(request.context)
        if state is not None:
            state[NANOCHAT_CORE_RESULTS_KEY] = results

        return EvaluationResult(
            evaluator=NANOCHAT_CORE_EVALUATOR,
            primary_metric="core_metric",
            metrics={"core_metric": core_metric},
            higher_is_better={"core_metric": True},
            metadata=_core_metadata(results),
        )


@contextlib.contextmanager
def _download_guard(data_path: str):
    """Serialize dataset downloads to avoid concurrent corruption."""
    os.makedirs(data_path, exist_ok=True)
    lock_file = os.path.join(data_path, ".download.lock")
    lock_fd = None
    waited = False

    try:
        while True:
            try:
                lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except FileExistsError:
                if not waited:
                    LOGGER.info(
                        "Another process is preparing the dataset at %s. Waiting.",
                        data_path,
                    )
                    waited = True
                time.sleep(1)
        yield
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                os.remove(lock_file)
            except FileNotFoundError:
                pass


def _download_eval_bundle(url: str, data_path: str) -> None:
    """Download the CORE evaluation bundle from a URL if not already available."""
    url_parse = urlparse(url)
    file_name = os.path.join(data_path, url_parse.path.split("/")[-1])
    os.makedirs(data_path, exist_ok=True)
    sentinel = Path(f"{file_name}.complete")

    if sentinel.exists():
        return

    with _download_guard(data_path):
        if sentinel.exists():
            return

        LOGGER.info("Downloading CORE evaluation bundle from %s.", url)

        res = requests.get(url, stream=True, timeout=60)
        total_size = int(res.headers.get("Content-Length", 0))
        downloaded_size = 0

        with open(file_name, "wb+") as file:
            for chunk in res.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                downloaded_size += len(chunk)
                file.write(chunk)
                file.flush()
                if total_size:
                    sys.stdout.write(f"\r{100 * downloaded_size / total_size:.1f}%")
                    sys.stdout.flush()
            if total_size:
                sys.stdout.write("\n")

        # Unzip the compressed file just downloaded
        LOGGER.info("Decompressing the CORE evaluation bundle.")
        name, suffix = os.path.splitext(file_name)

        if file_name.endswith("tar.gz"):
            with tarfile.open(file_name, "r:gz") as tar:
                tar.extractall(data_path)
            os.remove(file_name)
        elif suffix == ".zip":
            LOGGER.info("Extracting %s to %s.", file_name, data_path)
            with zipfile.ZipFile(file_name, "r") as zip_ref:
                zip_ref.extractall(data_path)
            os.remove(file_name)
        elif suffix == ".gz":
            with gzip.open(file_name, "rb") as zipped_file:
                with open(name, "wb") as unzipped_file:
                    unzipped_file.write(zipped_file.read())
            os.remove(file_name)
        else:
            LOGGER.warning("Unknown compressed file type for %s.", file_name)

        sentinel.touch()
        LOGGER.info("CORE evaluation bundle downloaded and extracted successfully.")


def _resolve_bundle_paths(bundle_dir: str | Path | None) -> tuple[Path, Path, Path]:
    """Resolve the configuration, metadata, and dataset paths for CORE evaluation."""
    ensure_nanochat_importable()

    def _get_default_base_path() -> Path:
        """Get the default base path, trying nanochat first, then Plato's data directory."""
        try:
            from nanochat.common import get_base_dir  # pylint: disable=import-error

            return Path(get_base_dir())
        except (ImportError, OSError, PermissionError):
            plato_data_path = Config().params.get("data_path", "./runtime/data")
            path = Path(plato_data_path) / "nanochat"
            LOGGER.info("Using Plato data directory for CORE bundle: %s", path)
            return path

    # Determine base path
    if bundle_dir is not None:
        try:
            base_path = Path(bundle_dir).expanduser().resolve()
            LOGGER.info("Using bundle_dir from config: %s", base_path)
        except (OSError, PermissionError, ValueError) as exc:
            LOGGER.warning(
                "Cannot use bundle_dir '%s': %s. Using default location.",
                bundle_dir,
                exc,
            )
            base_path = _get_default_base_path()
    else:
        base_path = _get_default_base_path()

    # Ensure base path exists
    base_path.mkdir(parents=True, exist_ok=True)

    eval_bundle_dir = base_path / "eval_bundle"
    config_path = eval_bundle_dir / "core.yaml"
    data_dir = eval_bundle_dir / "eval_data"
    metadata_path = eval_bundle_dir / "eval_meta_data.csv"

    # Check if evaluation bundle exists, download if missing
    if not config_path.exists() or not data_dir.exists() or not metadata_path.exists():
        LOGGER.info(
            "CORE evaluation bundle not found at %s. Downloading automatically...",
            base_path,
        )
        _download_eval_bundle(EVAL_BUNDLE_URL, str(base_path))

        # Verify download succeeded
        if not config_path.exists():
            raise FileNotFoundError(
                f"CORE evaluation config not found at {config_path}. "
                "Ensure the Nanochat eval bundle is downloaded."
            )
        if not data_dir.exists():
            raise FileNotFoundError(
                f"CORE evaluation data directory not found at {data_dir}. "
                "Ensure the Nanochat eval bundle is downloaded."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"CORE evaluation metadata CSV not found at {metadata_path}."
            )

    return config_path, data_dir, metadata_path


def _load_core_tasks(config_path: Path) -> list[dict[str, Any]]:
    """Load task definitions from the CORE YAML config."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    tasks = config.get("icl_tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(
            f"No CORE tasks defined in {config_path}. Inspect the eval bundle."
        )
    return tasks


def _load_metadata(metadata_path: Path) -> dict[str, float]:
    """Load random baseline metadata for centering accuracy."""
    baseline_map: dict[str, float] = {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row.get("Eval Task")
            baseline = row.get("Random baseline")
            if label is None or baseline is None:
                continue
            try:
                baseline_map[label] = float(baseline)
            except ValueError:
                LOGGER.debug("Skipping malformed baseline row: %s", row)
    if not baseline_map:
        raise ValueError(
            f"Random baselines missing in {metadata_path}. Required for CORE metric."
        )
    return baseline_map


def _load_task_data(data_dir: Path, dataset_uri: str) -> list[dict[str, Any]]:
    """Load task dataset rows from newline-delimited JSON."""
    path = data_dir / dataset_uri
    if not path.exists():
        raise FileNotFoundError(
            f"CORE dataset shard '{dataset_uri}' missing under {data_dir}."
        )
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line.strip()) for line in handle if line.strip()]


def _resolve_tokenizer(model) -> Any:
    """Obtain a tokenizer compatible with Nanochat core evaluation."""
    tokenizer = getattr(model, "nanochat_tokenizer", None)
    if tokenizer is not None:
        return tokenizer

    ensure_nanochat_importable()
    from nanochat.tokenizer import get_tokenizer  # pylint: disable=import-error

    return get_tokenizer()


def _safe_evaluate_task(
    model, tokenizer, data, device, task_meta, label,
    evaluate_task_fn, evaluate_example_fn,
):
    """Wrap upstream ``evaluate_task`` so that examples whose tokenized
    prompts exceed the model's ``max_seq_len`` are gracefully skipped
    instead of triggering an ``AssertionError``.

    Returns the mean accuracy over successfully evaluated examples, or
    ``None`` if every example had to be skipped.
    """
    try:
        # Fast path: let the upstream function handle everything.
        return evaluate_task_fn(model, tokenizer, data, device, task_meta)
    except AssertionError:
        pass  # Fall through to per-example evaluation.

    # Slow / safe path: evaluate one example at a time, skipping failures.
    correct = 0
    evaluated = 0
    for idx in range(len(data)):
        try:
            is_correct = evaluate_example_fn(
                idx, model, tokenizer, data, device, task_meta
            )
            correct += int(is_correct)
            evaluated += 1
        except AssertionError:
            LOGGER.debug(
                "CORE task %s: example %d exceeds max_seq_len, skipping.",
                label,
                idx,
            )
    if evaluated == 0:
        return None
    LOGGER.info(
        "CORE task %s: evaluated %d/%d examples (skipped %d too-long).",
        label,
        evaluated,
        len(data),
        len(data) - evaluated,
    )
    return correct / evaluated


def run_core_evaluation(
    model: torch.nn.Module,
    *,
    tokenizer: Any | None = None,
    bundle_dir: str | Path | None = None,
    max_per_task: int = -1,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """
    Execute the CORE benchmark for the provided model.

    Args:
        model: Nanochat-style autoregressive model.
        tokenizer: Optional tokenizer; falls back to nanochat.tokenizer.get_tokenizer().
        bundle_dir: Optional base directory containing `eval_bundle/`.
        max_per_task: Optional cap on examples per task for quicker smoke tests (-1 = all).
        device: Device to run evaluation on. Defaults to the model's current device.

    Returns:
        Dictionary with `results`, `centered_results`, and `core_metric`.
    """
    ensure_nanochat_importable()
    from nanochat.core_eval import (  # pylint: disable=import-error
        evaluate_example,
        evaluate_task,
    )

    config_path, data_dir, metadata_path = _resolve_bundle_paths(bundle_dir)
    tasks = _load_core_tasks(config_path)
    baselines = _load_metadata(metadata_path)

    eval_tokenizer = tokenizer or _resolve_tokenizer(model)
    if eval_tokenizer is None:
        raise RuntimeError(
            "Nanochat CORE evaluation requires a tokenizer. "
            "Either attach `model.nanochat_tokenizer` or provide one explicitly."
        )

    if device is None:
        try:
            first_param = next(model.parameters())
            device = first_param.device
        except StopIteration:
            device = torch.device("cpu")
    if isinstance(device, str):
        device = torch.device(device)

    model_device = device
    model_was_training = model.training
    model = model.to(model_device)
    model.eval()

    results: dict[str, float] = {}
    centered_results: dict[str, float] = {}

    for task in tasks:
        label = task.get("label")
        if not label:
            LOGGER.debug("Skipping unnamed CORE task entry: %s", task)
            continue

        dataset_uri = task.get("dataset_uri")
        if not isinstance(dataset_uri, str):
            LOGGER.debug(
                "Skipping CORE task %s due to missing dataset_uri metadata.", task
            )
            continue

        task_meta = {
            "task_type": task.get("icl_task_type"),
            "dataset_uri": dataset_uri,
            "num_fewshot": task.get("num_fewshot", [0])[0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        start_time = time.perf_counter()

        data = _load_task_data(data_dir, dataset_uri)
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = _safe_evaluate_task(
            model, eval_tokenizer, data, model_device, task_meta, label,
            evaluate_task, evaluate_example,
        )
        if accuracy is None:
            # All examples were skipped (too long for model's max_seq_len).
            LOGGER.warning(
                "CORE task %s skipped: all examples exceed model max_seq_len.",
                label,
            )
            continue
        baseline = baselines.get(label, 0.0)
        centered = (accuracy - 0.01 * baseline) / (1.0 - 0.01 * baseline)

        results[label] = accuracy
        centered_results[label] = centered
        elapsed = time.perf_counter() - start_time
        LOGGER.info(
            "CORE task %s | accuracy %.4f | centered %.4f | %.2fs",
            label,
            accuracy,
            centered,
            elapsed,
        )

    if model_was_training:
        model.train()

    if not centered_results:
        raise RuntimeError("No CORE tasks were evaluated; check the eval bundle.")

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }
