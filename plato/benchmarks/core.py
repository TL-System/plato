"""
CORE benchmark implementation for evaluating language models.
Borrowed and adapted from: https://github.com/karpathy/nanochat
"""

import json
import logging
import os
import random
import time
from typing import Any

import pandas as pd
import torch
import yaml

from plato.benchmarks import base
from plato.benchmarks.core_helpers import core
from plato.config import Config


class Benchmark(base.Benchmark):
    """
    CORE benchmark - evaluates language models on the CORE suite.
    """

    def __init__(self):
        """
        Initialize CORE benchmark -- load benchmark tasks and data.
        """
        super().__init__()

        # These will be set externally before evaluate() is called
        self.model = None
        self.device = None
        self.tokenizer = None

        # Get configuration specific to CORE benchmark
        self.random_seed = getattr(Config().benchmark, "random_seed", 24)
        self.max_per_task = getattr(Config().benchmark, "max_per_task", -1)

        # Load benchmark tasks and datasets
        self._load_benchmark_data()

    def _load_benchmark_data(self):
        """
        Load CORE benchmark tasks and evaluation data.

        Downloads the evaluation bundle if not already present, then loads
        task configurations and data files.
        """
        # Get base directory and ensure eval_bundle is downloaded
        benchmark_base_dir = Config.params["benchmark_path"]

        # Download eval_bundle if not present
        if not os.path.exists(benchmark_base_dir):
            logging.info("CORE evaluation bundle not found. Downloading...")
            eval_bundle_url = (
                "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
            )
            Benchmark.download(eval_bundle_url, benchmark_base_dir)

        # Load benchmark configuration
        eval_bundle_dir = os.path.join(benchmark_base_dir, "eval_bundle")
        config_path = os.path.join(eval_bundle_dir, "core.yaml")
        self.eval_meta_data_path = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
        self.data_base_path = os.path.join(eval_bundle_dir, "eval_data")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.tasks = config["icl_tasks"]
        self.eval_metadata = pd.read_csv(self.eval_meta_data_path)

    def evaluate(self) -> dict[str, Any]:
        """
        Evaluate the model on all CORE tasks.

        Returns:
            Dictionary containing:
                - 'results': per-task accuracies
                - 'centered_results': normalized scores
                - 'core_metric': overall CORE score
        """

        if self.model is None:
            raise RuntimeError("Trainer has no model - cannot run benchmark")

        if self.tokenizer is None:
            raise RuntimeError("Trainer has no tokenizer - cannot run benchmark")

        results = {}
        centered_results = {}

        # Set model to eval mode
        self.model.eval()

        with torch.no_grad():
            for task in self.tasks:
                start_time = time.time()
                label = task["label"]

                task_meta = {
                    "task_type": task["icl_task_type"],
                    "dataset_uri": task["dataset_uri"],
                    "num_fewshot": task["num_fewshot"][0],
                    "continuation_delimiter": task.get("continuation_delimiter", " "),
                }

                logging.info(
                    "Evaluating task: %s (%d-shot, type: %s)",
                    label,
                    task_meta["num_fewshot"],
                    task_meta["task_type"],
                )

                # Load data for this task (matching evaluate_model.py pattern)
                data_path = os.path.join(self.data_base_path, task_meta["dataset_uri"])
                with open(data_path, "r") as f:
                    data = [json.loads(line.strip()) for line in f]

                # Shuffle the data for reproducibility (matching evaluate_model.py)
                shuffle_rng = random.Random(self.random_seed)
                shuffle_rng.shuffle(data)

                # Crop data if max_per_task is specified
                if self.max_per_task > 0:
                    data = data[: self.max_per_task]

                # Run evaluation using existing core_eval logic
                accuracy = core.evaluate_task(
                    self.model,  # Model in CUDA memory from trainer
                    self.tokenizer,  # Tokenizer from trainer
                    data,
                    self.device,
                    task_meta,
                )

                results[label] = accuracy

                # Compute centered result (normalized by random baseline)
                row = self.eval_metadata[self.eval_metadata["Eval Task"] == label]
                random_baseline = row["Random baseline"].values[0]
                centered = (accuracy - 0.01 * random_baseline) / (
                    1.0 - 0.01 * random_baseline
                )
                centered_results[label] = centered

                elapsed = time.time() - start_time
                logging.info(
                    "accuracy: %.4f | centered: %.4f | time: %.2fs",
                    accuracy,
                    centered,
                    elapsed,
                )

        # Compute overall CORE metric
        core_metric = sum(centered_results.values()) / len(centered_results)

        return {
            "results": results,
            "centered_results": centered_results,
            "core_metric": core_metric,
        }

    def get_formatted_result(self, evaluation_result: dict[str, Any]) -> str:
        """
        Format the evaluation results for display.

        Args:
            evaluation_result: The dictionary returned by the evaluate() method.
        Returns:
            A formatted string summarizing the results.
        """
        results = evaluation_result["results"]
        centered_results = evaluation_result["centered_results"]
        core_metric = evaluation_result["core_metric"]

        result_lines = [f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}"]
        for task, acc in results.items():
            centered = centered_results[task]
            result_lines.append(f"{task:<35}, {acc:<10.6f}, {centered:<10.6f}")
        result_lines.append(
            f"{'Overall CORE Metric':<35}, {'':<10}, {core_metric:<10.6f}\n"
        )

        return "\n".join(result_lines)
