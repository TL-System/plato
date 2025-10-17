"""Tests for the FedNova example training strategy."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from plato.config import Config

_TESTS_ROOT = Path(__file__).resolve().parent
_FEDNOVA_CLIENT_PATH = (
    _TESTS_ROOT.parent.parent
    / "examples"
    / "server_aggregation"
    / "fednova"
    / "fednova_client.py"
)
_FEDNOVA_SPEC = importlib.util.spec_from_file_location(
    "fednova_client_module", _FEDNOVA_CLIENT_PATH
)
fednova_client = importlib.util.module_from_spec(_FEDNOVA_SPEC)
assert _FEDNOVA_SPEC.loader is not None
_FEDNOVA_SPEC.loader.exec_module(fednova_client)


def test_report_contains_epochs_for_constant_pattern(temp_config):
    """With a fixed epoch configuration the report should record the value."""
    Config().trainer = Config().trainer._replace(epochs=2)

    strategy = fednova_client.FedNovaTrainingStrategy()
    context = SimpleNamespace(client_id=1, state={})

    mock_report = SimpleNamespace(num_samples=10)
    mock_weights = {"weights": 1}
    async_mock = AsyncMock(return_value=(mock_report, mock_weights))

    with patch.object(
        fednova_client.DefaultTrainingStrategy,
        "train",
        new=async_mock,
    ) as mock_train:
        report, weights = asyncio.run(strategy.train(context))

    mock_train.assert_awaited_once()
    assert weights is mock_weights
    assert hasattr(report, "epochs")
    assert report.epochs == 2


def test_uniform_random_assigns_epochs_and_updates_config(temp_config):
    """The uniform random pattern should select epochs and update the config."""
    strategy = fednova_client.FedNovaTrainingStrategy()
    context = SimpleNamespace(client_id=3, state={})

    mock_report = SimpleNamespace(num_samples=25)
    async_mock = AsyncMock(return_value=(mock_report, None))

    original_algorithm = Config.algorithm
    original_trainer = Config().trainer
    Config.algorithm = SimpleNamespace(pattern="uniform_random", max_local_epochs=6)

    try:
        with (
            patch.object(
                fednova_client.DefaultTrainingStrategy,
                "train",
                new=async_mock,
            ) as mock_train,
            patch.object(
                fednova_client.np.random,
                "randint",
                return_value=5,
            ) as mock_randint,
        ):
            report, _ = asyncio.run(strategy.train(context))

    finally:
        Config.algorithm = original_algorithm
        Config.trainer = original_trainer

    mock_train.assert_awaited_once()
    mock_randint.assert_called_once_with(2, 7)
    assert report.epochs == 5
    assert Config().trainer.epochs == 5
