"""
Adapter utilities to run Nanochat's CORE evaluation benchmark within Plato.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Nanochat CORE evaluation requires PyTorch. "
        "Install the `nanochat` extra (includes torch)."
    ) from exc

import yaml

from plato.utils.third_party import ThirdPartyImportError, ensure_nanochat_importable

LOGGER = logging.getLogger(__name__)


def _resolve_bundle_paths(bundle_dir: str | Path | None) -> tuple[Path, Path, Path]:
    """Resolve the configuration, metadata, and dataset paths for CORE evaluation."""
    ensure_nanochat_importable()
    from nanochat.common import get_base_dir  # pylint: disable=import-error

    if bundle_dir is None:
        base_path = Path(get_base_dir())
    else:
        base_path = Path(bundle_dir).expanduser().resolve()

    eval_bundle_dir = base_path / "eval_bundle"
    config_path = eval_bundle_dir / "core.yaml"
    data_dir = eval_bundle_dir / "eval_data"
    metadata_path = eval_bundle_dir / "eval_meta_data.csv"

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
    from nanochat.core_eval import evaluate_task  # pylint: disable=import-error

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

        task_meta = {
            "task_type": task.get("icl_task_type"),
            "dataset_uri": task.get("dataset_uri"),
            "num_fewshot": task.get("num_fewshot", [0])[0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        start_time = time.perf_counter()

        data = _load_task_data(data_dir, task_meta["dataset_uri"])
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, eval_tokenizer, data, model_device, task_meta)
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
