"""
Helpers for integration smoke tests to provision configs and runtime context.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import yaml

from plato.config import Config


def build_minimal_config(
    *,
    rounds: int = 1,
    clients_per_round: int = 1,
    total_clients: int = 2,
    model_name: str = "lenet5",
    trainer_type: str = "basic",
    client_type: str = "simple",
) -> dict:
    """Create a minimal config dictionary suitable for smoke tests."""
    config_text = textwrap.dedent(
        f"""
        clients:
            type: {client_type}
            total_clients: {total_clients}
            per_round: {clients_per_round}
            do_test: false

        server:
            address: 127.0.0.1
            port: 8000
            random_seed: 1
            simulate_wall_time: true

        data:
            datasource: toy
            partition_size: 4
            sampler: iid
            random_seed: 1

        trainer:
            type: {trainer_type}
            rounds: {rounds}
            epochs: 1
            batch_size: 2
            optimizer: SGD
            model_name: {model_name}

        algorithm:
            type: fedavg

        parameters:
            optimizer:
                lr: 0.01
                momentum: 0.0
                weight_decay: 0.0
        """
    )
    return yaml.safe_load(config_text)


@contextlib.contextmanager
def configure_environment(config_dict: dict):
    """
    Context manager that writes the config to disk and initialises Config singleton.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.yml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config_dict, handle)

        Config._instance = None  # reset singleton
        Config.params = {}

        previous_env = os.environ.get("config_file")
        previous_argv = sys.argv[:]
        os.environ["config_file"] = str(config_path)
        sys.argv = [previous_argv[0]] if previous_argv else ["pytest"]

        try:
            config = Config()
        finally:
            if previous_env is None:
                os.environ.pop("config_file", None)
            else:
                os.environ["config_file"] = previous_env
            sys.argv = previous_argv

        Config.args.id = 0
        model_dir = Path(tmp_dir) / "models"
        ckpt_dir = Path(tmp_dir) / "checkpoints"
        results_dir = Path(tmp_dir) / "results"

        for directory in (model_dir, ckpt_dir, results_dir):
            directory.mkdir(parents=True, exist_ok=True)

        Config.params["base_path"] = tmp_dir
        Config.params["model_path"] = str(model_dir)
        Config.params["checkpoint_path"] = str(ckpt_dir)
        Config.params["result_path"] = str(results_dir)

        yield config

        Config._instance = None


def async_run(coro):
    """Utility to execute the coroutine using asyncio.run (Python 3.7+)."""
    return asyncio.run(coro)
