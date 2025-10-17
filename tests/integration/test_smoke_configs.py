"""
Integration smoke tests covering minimal client-server orchestration.
"""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from tests.integration.utils import (
    async_run,
    build_minimal_config,
    configure_environment,
)


class MNISTSmokeDatasource:
    """Datasource returning image-shaped tensors for LeNet smoke tests."""

    def __init__(self, train_size: int = 4, test_size: int = 2):
        self._train = TensorDataset(
            torch.randn(train_size, 1, 28, 28),
            torch.randint(0, 10, (train_size,)),
        )
        self._test = TensorDataset(
            torch.randn(test_size, 1, 28, 28),
            torch.randint(0, 10, (test_size,)),
        )

    def num_train_examples(self):
        return len(self._train)

    def get_train_set(self):
        return self._train

    def get_test_set(self):
        return self._test


@pytest.mark.integration
def test_fedavg_lenet5_smoke(monkeypatch):
    """End-to-end smoke test for a minimal FedAvg run."""
    config = build_minimal_config(
        rounds=1,
        clients_per_round=1,
        model_name="lenet5",
    )
    with configure_environment(config):
        datasources_registry = import_module("plato.datasources.registry")
        processor_registry = import_module("plato.processors.registry")

        fake_datasource = MNISTSmokeDatasource()

        monkeypatch.setattr(
            datasources_registry,
            "get",
            lambda *args, **kwargs: fake_datasource,
        )
        monkeypatch.setattr(
            processor_registry,
            "get",
            lambda *args, **kwargs: (None, None),
        )

        server_mod = import_module("plato.servers.fedavg")
        server = server_mod.Server()
        server.configure()

        # Build fake updates to trigger aggregation without real clients.
        trainer = server.trainer
        weights = {
            name: tensor.clone() for name, tensor in trainer.model.state_dict().items()
        }
        update = SimpleNamespace(
            client_id=1,
            report=SimpleNamespace(
                num_samples=1,
                accuracy=0.5,
                processing_time=0.1,
                comm_time=0.1,
                training_time=0.1,
            ),
            payload=weights,
        )
        server.updates = [update]
        server.current_round = 0
        server.context.current_round = 0

        async_run(server._process_reports())
        assert server.accuracy >= 0


@pytest.mark.integration
def test_split_learning_smoke(monkeypatch):
    """Smoke test for split-learning trainer orchestrating gradients."""
    config = build_minimal_config(
        trainer_type="split_learning",
        rounds=1,
        clients_per_round=1,
        model_name="split_cnn",
    )

    with configure_environment(config):
        trainer_mod = import_module("plato.trainers.split_learning")
        trainer = trainer_mod.Trainer(model=lambda: SimpleNamespace())
        trainer.context.client_id = 0
        trainer.gradients = []
        trainer.cut_layer_grad = None

        trainer.callback_handler.call_event("on_train_run_start", trainer, {})
        trainer.callback_handler.call_event(
            "on_train_run_end", trainer, {"model_name": "split_cnn"}
        )
        assert trainer.context.state["trainer"] is trainer


@pytest.mark.integration
def test_mpc_training_smoke(monkeypatch):
    """Smoke test ensuring MPC training strategy registers sample counts."""
    config = build_minimal_config(
        trainer_type="basic",
        rounds=1,
        clients_per_round=1,
        model_name="lenet5",
        client_type="mpc",
    )

    with configure_environment(config):
        round_store_calls = []

        class DummyRoundStore:
            def record_client_samples(self, client_id, num_samples):
                round_store_calls.append((client_id, num_samples))

        dummy_store = DummyRoundStore()
        strategy_mod = import_module("plato.clients.strategies.mpc")
        strategy = strategy_mod.MPCTrainingStrategy(dummy_store)

        async def fake_train(self, context):
            report = SimpleNamespace(num_samples=3)
            return report, {}

        defaults = import_module("plato.clients.strategies.defaults")
        monkeypatch.setattr(
            defaults.DefaultTrainingStrategy,
            "train",
            fake_train,
            raising=False,
        )

        context_mod = import_module("plato.clients.strategies.base")
        client_context = context_mod.ClientContext()
        client_context.client_id = 1

        async_run(strategy.train(client_context))
        assert round_store_calls == [(1, 3)]
