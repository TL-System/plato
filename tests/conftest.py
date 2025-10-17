"""Pytest fixtures shared across test modules."""

import os
import sys
import textwrap
from pathlib import Path

import pytest

from plato.config import Config


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    """Provide an isolated configuration for tests relying on Config."""

    config_text = textwrap.dedent(
        """
        clients:
            type: simple
            total_clients: 2
            per_round: 2
            do_test: false

        server:
            address: 127.0.0.1
            port: 8000

        data:
            datasource: toy
            partition_size: 4
            sampler: iid
            random_seed: 1

        trainer:
            type: basic
            rounds: 1
            epochs: 1
            batch_size: 2
            optimizer: SGD
            model_name: toy_model

        algorithm:
            type: fedavg

        parameters:
            optimizer:
                lr: 0.1
                momentum: 0.0
                weight_decay: 0.0
        """
    ).strip()

    config_path = tmp_path / "config.yml"
    config_path.write_text(config_text, encoding="utf-8")

    monkeypatch.setenv("config_file", str(config_path))
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])

    # Reset the Config singleton so each test gets a clean instance.
    Config._instance = None

    config = Config()

    # Redirect model and checkpoint directories into the temp folder.
    base_path = Path(tmp_path)
    Config.params["base_path"] = str(base_path)
    Config.params["model_path"] = str(base_path / "models")
    Config.params["checkpoint_path"] = str(base_path / "checkpoints")
    os.makedirs(Config.params["model_path"], exist_ok=True)
    os.makedirs(Config.params["checkpoint_path"], exist_ok=True)

    yield config

    # Tear down the singleton so subsequent tests can configure a new instance.
    Config._instance = None
