"""
Helpers for integration smoke tests to provision configs and runtime context.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator

from plato.config import Config
from plato.utils import toml_writer


@contextlib.contextmanager
def isolated_config_state() -> Iterator[None]:
    """Detach test configuration and restore the caller's class state on exit."""

    def is_state(name, value):
        return not name.startswith("__") and not (
            callable(value) or isinstance(value, (classmethod, staticmethod))
        )

    previous = {
        name: value for name, value in vars(Config).items() if is_state(name, value)
    }
    try:
        for name in previous:
            delattr(Config, name)
        Config._instance = None
        Config._cli_overrides = {}
        Config.client_sleep_times = None
        yield
    finally:
        for name, value in list(vars(Config).items()):
            if is_state(name, value):
                delattr(Config, name)
        for name, value in previous.items():
            setattr(Config, name, value)


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
    return {
        "clients": {
            "type": client_type,
            "total_clients": total_clients,
            "per_round": clients_per_round,
            "do_test": False,
        },
        "server": {
            "address": "127.0.0.1",
            "port": 8000,
            "random_seed": 1,
            "simulate_wall_time": True,
        },
        "data": {
            "datasource": "toy",
            "partition_size": 4,
            "sampler": "iid",
            "random_seed": 1,
        },
        "trainer": {
            "type": trainer_type,
            "rounds": rounds,
            "epochs": 1,
            "batch_size": 2,
            "optimizer": "SGD",
            "model_name": model_name,
        },
        "algorithm": {"type": "fedavg"},
        "parameters": {
            "optimizer": {
                "lr": 0.01,
                "momentum": 0.0,
                "weight_decay": 0.0,
            }
        },
    }


@contextlib.contextmanager
def configure_environment(
    config_dict: dict, *, runtime_root: Path | None = None
) -> Iterator[Config]:
    """
    Context manager that writes the config to disk and initialises Config singleton.
    """
    with tempfile.TemporaryDirectory() as tmp_dir, isolated_config_state():
        root = runtime_root if runtime_root is not None else Path(tmp_dir)
        root = root.resolve()
        config_path = root / "config.toml"
        config_data = copy.deepcopy(config_dict)
        # Absolute paths in input configs must not escape the test runtime root.
        config_data.setdefault("data", {})["data_path"] = "data"
        server = config_data.setdefault("server", {})
        server.update(
            model_path="models", checkpoint_path="checkpoints", mpc_data_path="mpc"
        )
        config_data.setdefault("results", {})["result_path"] = "results"
        toml_writer.dump(config_data, config_path)
        previous_env = os.environ.get("config_file")
        previous_argv = sys.argv
        os.environ["config_file"] = str(config_path)
        program = previous_argv[0] if previous_argv else "pytest"
        sys.argv = [program, "-b", str(root), "--cpu"]

        try:
            config = Config()
            Config.args.id = 0
            Path(Config.params["data_path"]).mkdir(parents=True, exist_ok=True)
            yield config
        finally:
            if previous_env is None:
                os.environ.pop("config_file", None)
            else:
                os.environ["config_file"] = previous_env
            sys.argv = previous_argv


@contextlib.contextmanager
def configure_environment_from_path(config_path: Path):
    """
    Context manager that initialises Config singleton from an existing config.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        Config._instance = None  # reset singleton
        Config.params = {}

        previous_env = os.environ.get("config_file")
        previous_argv = sys.argv[:]
        os.environ["config_file"] = str(config_path)
        sys.argv = [
            previous_argv[0] if previous_argv else "pytest",
            "--base",
            tmp_dir,
        ]

        try:
            config = Config()
            yield config
        finally:
            if previous_env is None:
                os.environ.pop("config_file", None)
            else:
                os.environ["config_file"] = previous_env
            sys.argv = previous_argv
            Config._instance = None


def async_run(coro):
    """Utility to execute the coroutine using asyncio.run (Python 3.7+)."""
    return asyncio.run(coro)
