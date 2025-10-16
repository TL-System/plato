import asyncio
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

_TESTS_ROOT = Path(__file__).resolve().parent
os.environ["config_file"] = str(_TESTS_ROOT / "TestsConfig" / "fedavg_tests.yml")
sys.argv = [sys.argv[0]]

from plato.config import Config

_FEDNOVA_CLIENT_PATH = (
    _TESTS_ROOT.parent
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


class FedNovaTrainingStrategyTests(unittest.TestCase):
    def setUp(self):
        Config().trainer = Config().trainer._replace(epochs=2)

    def test_report_contains_epochs_for_constant_pattern(self):
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
        self.assertIs(weights, mock_weights)
        self.assertTrue(hasattr(report, "epochs"))
        self.assertEqual(report.epochs, 2)

    def test_uniform_random_assigns_epochs_and_updates_config(self):
        strategy = fednova_client.FedNovaTrainingStrategy()
        context = SimpleNamespace(client_id=3, state={})

        mock_report = SimpleNamespace(num_samples=25)
        async_mock = AsyncMock(return_value=(mock_report, None))

        original_algorithm = Config.algorithm
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

        mock_train.assert_awaited_once()
        mock_randint.assert_called_once_with(2, 7)
        self.assertEqual(report.epochs, 5)
        self.assertEqual(Config().trainer.epochs, 5)


if __name__ == "__main__":
    unittest.main()
