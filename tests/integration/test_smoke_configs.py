"""
Integration smoke tests covering minimal client-server orchestration.
"""

from __future__ import annotations

from importlib import import_module, reload
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch.utils.data import TensorDataset

from plato.mpc.round_store import RoundInfoStore
from tests.integration.utils import (
    async_run,
    build_minimal_config,
    configure_environment,
    configure_environment_from_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class MNISTSmokeDatasource:
    """Datasource returning image-shaped tensors for LeNet smoke tests."""

    def __init__(self, train_size: int = 4, test_size: int = 2):
        generator = torch.Generator().manual_seed(13)
        self._train = TensorDataset(
            torch.randn(train_size, 1, 28, 28, generator=generator),
            torch.randint(0, 10, (train_size,), generator=generator),
        )
        self._test = TensorDataset(
            torch.randn(test_size, 1, 28, 28, generator=generator),
            torch.randint(0, 10, (test_size,), generator=generator),
        )

    def num_train_examples(self):
        return len(self._train)

    def get_train_set(self):
        return self._train

    def get_test_set(self):
        return self._test


LOCAL_STATE_PAYLOAD_KEYS = {
    "optimizer_state",
    "scheduler_state",
    "trainer_state",
    "local_metadata",
    "metadata",
    "global_step",
    "local_optimizer_steps",
    "_optimizer_state_input_filename",
    "_optimizer_state_output_filename",
}


def _model_weight_payload(payload, model):
    """Assert that a client payload contains model weights only."""
    model_state = model.state_dict()

    assert isinstance(payload, dict)
    assert set(payload) == set(model_state)
    assert LOCAL_STATE_PAYLOAD_KEYS.isdisjoint(payload)
    assert all(torch.is_tensor(value) for value in payload.values())


def _shifted_payload(weights, amount):
    """Build a fake client model payload shifted from the server baseline."""
    shifted = {}
    for name, value in weights.items():
        shifted[name] = value.clone()
        if torch.is_floating_point(shifted[name]):
            shifted[name] = shifted[name] + amount
    return shifted


def _server_update(client_id, payload):
    """Build a minimal server update carrying model weights."""
    return SimpleNamespace(
        client_id=client_id,
        report=SimpleNamespace(
            client_id=client_id,
            num_samples=1,
            accuracy=0.5,
            processing_time=0.1,
            comm_time=0.1,
            training_time=0.1,
            type="weights",
        ),
        payload=payload,
    )


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
        assert trainer is not None
        model = trainer.model
        assert model is not None
        weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
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
def test_diloco_lenet5_smoke_config_contract_loads():
    """Smoke config should load the faithful DiLoCo contract."""
    config_path = REPO_ROOT / "configs" / "MNIST" / "diloco_lenet5_smoke.toml"

    with configure_environment_from_path(config_path) as config:
        assert config.server.type == "diloco"
        assert config.algorithm.type == "fedavg"
        assert config.trainer.local_steps_per_round == 2
        assert config.trainer.preserve_optimizer_state is True
        assert config.trainer.optimizer == "AdamW"
        assert config.server.diloco.outer_optimizer == "nesterov"
        assert config.server.diloco.outer_learning_rate == 0.7
        assert config.server.diloco.outer_momentum == 0.9
        assert config.server.diloco.aggregation_weighting == "uniform"
        assert config.server.diloco.apply_outer_optimizer_to == "parameters"

        server_registry = reload(import_module("plato.servers.registry"))
        diloco_server = import_module("plato.servers.diloco")
        diloco_aggregation = import_module("plato.servers.strategies.aggregation")

        server = server_registry.get()

        assert isinstance(server, diloco_server.Server)
        assert isinstance(
            server.aggregation_strategy,
            diloco_aggregation.DiLoCoAggregationStrategy,
        )


@pytest.mark.integration
def test_diloco_lenet5_smoke_config_runs_faithful_path(monkeypatch):
    """Exact DiLoCo smoke config exercises local work and outer aggregation."""
    config_path = REPO_ROOT / "configs" / "MNIST" / "diloco_lenet5_smoke.toml"

    with configure_environment_from_path(config_path) as config:
        datasources_registry = import_module("plato.datasources.registry")
        processor_registry = import_module("plato.processors.registry")
        server_registry = reload(import_module("plato.servers.registry"))
        client_mod = import_module("plato.clients.simple")
        config_mod = import_module("plato.config")
        diloco_server = import_module("plato.servers.diloco")
        fedavg_algorithm = import_module("plato.algorithms.fedavg")
        diloco_aggregation = import_module("plato.servers.strategies.aggregation")

        fake_datasource = MNISTSmokeDatasource(train_size=32, test_size=4)
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

        server = server_registry.get()
        server.configure()

        assert isinstance(server, diloco_server.Server)
        assert isinstance(server.algorithm, fedavg_algorithm.Algorithm)
        assert isinstance(
            server.aggregation_strategy,
            diloco_aggregation.DiLoCoAggregationStrategy,
        )
        assert config.server.type == "diloco"
        assert config.algorithm.type == "fedavg"
        assert config.trainer.local_steps_per_round == 2
        assert config.trainer.preserve_optimizer_state is True
        assert config.data.sampler == "iid"

        client = client_mod.Client()
        client.client_id = 1
        client._context.client_id = 1
        client.current_round = 1
        client._context.current_round = 1
        client._load_data()
        client.configure()
        client._allocate_data()
        client._load_payload(server.algorithm.extract_weights())

        train_config = config.trainer._asdict()
        train_config["run_id"] = config_mod.Config.params["run_id"]
        client.trainer.current_round = client.current_round
        client.trainer.train_model(train_config, client.trainset, client.sampler)
        payload = client.algorithm.extract_weights()

        assert client.sampler.num_samples() == config.data.partition_size
        assert client.trainer.context.state["local_steps_per_round"] == 2
        assert client.trainer.context.state["local_optimizer_steps"] == 2
        assert client.trainer.current_epoch == 1
        assert client.client_id in client.trainer._preserved_optimizer_states
        assert client.trainer._preserved_optimizer_states[client.client_id][
            "optimizer_state"
        ]["state"]
        _model_weight_payload(payload, client.trainer.model)

        # Small-H mid-epoch stopping and round-aware sampler streaming are covered
        # in TestComposableTrainerLocalSteps; this integration path verifies the
        # exact smoke config enables those runtime flags with the supported sampler.
        baseline = server.algorithm.extract_weights()
        trainable_name = next(iter(dict(server.trainer.model.named_parameters())))
        server.updates = [
            _server_update(1, _shifted_payload(baseline, 1.0)),
            _server_update(2, _shifted_payload(baseline, 3.0)),
        ]
        server.current_round = 1
        server.context.current_round = 1

        delta_calls = []
        aggregate_deltas = server.aggregation_strategy.aggregate_deltas

        async def record_delta_aggregation(updates, deltas_received, context):
            delta_calls.append((updates, deltas_received))
            return await aggregate_deltas(updates, deltas_received, context)

        monkeypatch.setattr(
            server.aggregation_strategy,
            "aggregate_deltas",
            record_delta_aggregation,
        )

        async_run(server._process_reports())

        updated = server.algorithm.extract_weights()
        ordinary_fedavg_value = baseline[trainable_name] + 2.0
        faithful_diloco_value = baseline[trainable_name] + 2.0 * 1.9 * 0.7

        assert len(delta_calls) == 1
        assert len(delta_calls[0][1]) == 2
        assert not torch.allclose(updated[trainable_name], ordinary_fedavg_value)
        assert torch.allclose(updated[trainable_name], faithful_diloco_value)


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
        round_store = cast(RoundInfoStore, dummy_store)
        strategy_mod = import_module("plato.clients.strategies.mpc")
        strategy = strategy_mod.MPCTrainingStrategy(round_store)

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
