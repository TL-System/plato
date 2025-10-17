"""Pytest fixtures shared across test modules."""

import os
import sys
import textwrap
from pathlib import Path

import pytest

from plato.config import Config
from tests.test_utils.fakes import (
    FakeDatasource,
    FakeModel,
    IdentityLifecycleStrategy,
    InMemoryReportingStrategy,
    NoOpCommunicationStrategy,
    RecordingPayloadStrategy,
    StaticTrainingStrategy,
    WeightedAverageAggregation,
)


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
    if getattr(Config, "args", None) is not None and Config.args.id is None:
        Config.args.id = 1

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
    if getattr(Config, "args", None) is not None:
        Config.args.id = None


@pytest.fixture
def fake_model_cls():
    """Return the lightweight fake model class for composing components."""
    return FakeModel


@pytest.fixture
def fake_datasource_cls():
    """Return the fake datasource class to create deterministic datasets."""
    return FakeDatasource


@pytest.fixture
def fake_training_strategy():
    """Instantiate a training strategy that skips optimisation."""
    return StaticTrainingStrategy()


@pytest.fixture
def fake_lifecycle_strategy(fake_datasource_cls):
    """Lifecycle strategy that injects fake datasource/trainer components."""
    return IdentityLifecycleStrategy(datasource_factory=fake_datasource_cls)


@pytest.fixture
def fake_reporting_strategy():
    """Reporting strategy storing the most recent report in memory."""
    return InMemoryReportingStrategy()


@pytest.fixture
def fake_communication_strategy():
    """Communication strategy that records outbound artefacts."""
    return NoOpCommunicationStrategy()


@pytest.fixture
def recording_payload_strategy():
    """Payload strategy that records lifecycle events for assertions."""
    return RecordingPayloadStrategy()


@pytest.fixture
def fake_aggregation_strategy():
    """Aggregation strategy performing a simple weighted average."""
    return WeightedAverageAggregation()
