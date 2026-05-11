"""End-to-end smoke tests for the strategy-based client runtime."""

import asyncio
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from plato.algorithms import fedavg
from plato.clients import simple
from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies import AdamWOptimizerStrategy, StepLRSchedulerStrategy
from tests.test_utils.fakes import NoOpCommunicationStrategy

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


class ToyDataset(Dataset):
    """Simple dataset with deterministic feature-label pairs."""

    def __init__(self, length: int = 8, input_dim: int = 4):
        torch.manual_seed(42)
        self.inputs = torch.randn(length, input_dim)
        self.labels = torch.randint(0, 2, (length,))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


@dataclass
class ToyDatasource:
    """Datasource compatible with the default lifecycle strategy."""

    train_length: int = 8
    test_length: int = 4

    def __post_init__(self):
        self._train = ToyDataset(self.train_length)
        self._test = ToyDataset(self.test_length)

    def num_train_examples(self):
        return len(self._train)

    def get_train_set(self):
        return self._train

    def get_test_set(self):
        return self._test


def _build_client(trainer=ComposableTrainer):
    """Instantiate a client wired with custom model, datasource, and trainer."""
    return simple.Client(
        model=torch.nn.Linear(4, 2),
        datasource=ToyDatasource,
        trainer=trainer,
        algorithm=lambda trainer: fedavg.Algorithm(trainer),
    )


def _build_stateful_trainer(model=None, callbacks=None):
    """Build a trainer whose local optimizer and scheduler state is non-empty."""
    return ComposableTrainer(
        model=model,
        callbacks=callbacks,
        optimizer_strategy=AdamWOptimizerStrategy(lr=0.01),
        lr_scheduler_strategy=StepLRSchedulerStrategy(step_size=1, gamma=0.5),
    )


def _configure_one_round_client(client):
    """Prepare a client for a deterministic single training round."""
    client.client_id = 1
    client._context.client_id = 1
    client.current_round = 1
    client._context.current_round = 1

    client._load_data()
    client.configure()
    client._allocate_data()


def _disable_payload_processors(client):
    """Keep the test focused on decoded client-server model payload contents."""
    client.inbound_processor = None
    client.outbound_processor = None
    client._context.inbound_processor = None
    client._context.outbound_processor = None


def _assert_model_weight_payload(payload, model):
    """Assert that an outbound payload contains exactly model state tensors."""
    model_state = model.state_dict()

    assert isinstance(payload, dict)
    assert set(payload) == set(model_state)
    assert LOCAL_STATE_PAYLOAD_KEYS.isdisjoint(payload)
    assert all(torch.is_tensor(value) for value in payload.values())

    for name, expected in model_state.items():
        assert torch.equal(payload[name], expected)


def _assert_preserved_state_is_local(trainer, client_id):
    """Assert optimizer and scheduler persistence exists only in trainer state."""
    state = trainer._preserved_optimizer_states[client_id]

    assert state["optimizer_state"]["state"]
    assert state["scheduler_state"] is not None
    assert state["scheduler_state"]["last_epoch"] >= 1
    assert state["scheduler_state"]["_step_count"] >= 2


def test_simple_client_trains_with_default_strategies(temp_config):
    """A simple client should complete one training round using the strategy stack."""
    Config().trainer = Config().trainer._replace(epochs=1, batch_size=2)

    client = _build_client()

    _configure_one_round_client(client)

    report, payload = asyncio.run(client._train())

    assert report.client_id == 1
    # With partition_size=4 each client receives four samples.
    assert report.num_samples == 4
    _assert_model_weight_payload(payload, client.trainer.model)


def test_simple_client_payload_excludes_local_state_when_persistence_enabled(
    temp_config,
):
    """FedAvg/DiLoCo client payloads stay model-only with local persistence."""
    Config.params["run_id"] = "client-payload-in-process"
    Config().trainer = Config().trainer._replace(
        epochs=1,
        batch_size=2,
        preserve_optimizer_state=True,
    )
    client = _build_client(trainer=_build_stateful_trainer)
    client._configure_composable(
        lifecycle_strategy=client.lifecycle_strategy,
        payload_strategy=client.payload_strategy,
        training_strategy=client.training_strategy,
        reporting_strategy=client.reporting_strategy,
        communication_strategy=NoOpCommunicationStrategy(),
    )
    _configure_one_round_client(client)
    _disable_payload_processors(client)

    server_payload = client.algorithm.extract_weights()
    asyncio.run(client._handle_payload(server_payload))

    sent_payload = client._context.state["sent_payloads"][-1]
    _assert_preserved_state_is_local(client.trainer, client.client_id)
    _assert_model_weight_payload(sent_payload, client.trainer.model)


def test_simple_client_subprocess_payload_excludes_local_state_sidecar(
    temp_config, monkeypatch, tmp_path
):
    """Subprocess persistence uses a sidecar without changing server payloads."""
    model_path = Path(tmp_path) / "models" / "pretrained"
    checkpoint_path = Path(tmp_path) / "checkpoints"
    model_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    Config.params["model_path"] = str(model_path)
    Config.params["checkpoint_path"] = str(checkpoint_path)
    Config.params["run_id"] = "client-payload-subprocess"
    monkeypatch.setattr(sys, "argv", [sys.argv[0], "-b", str(tmp_path)])
    Config().trainer = Config().trainer._replace(
        epochs=1,
        batch_size=2,
        max_concurrency=1,
        preserve_optimizer_state=True,
    )
    client = _build_client(trainer=_build_stateful_trainer)
    client._configure_composable(
        lifecycle_strategy=client.lifecycle_strategy,
        payload_strategy=client.payload_strategy,
        training_strategy=client.training_strategy,
        reporting_strategy=client.reporting_strategy,
        communication_strategy=NoOpCommunicationStrategy(),
    )
    _configure_one_round_client(client)
    _disable_payload_processors(client)

    server_payload = client.algorithm.extract_weights()
    asyncio.run(client._handle_payload(server_payload))

    sent_payload = client._context.state["sent_payloads"][-1]
    state_path = Path(
        Config.params["model_path"]
    ) / client.trainer._optimizer_state_filename(Config.params["run_id"])
    with state_path.open("rb") as state_file:
        sidecar_state = pickle.load(state_file)

    _assert_preserved_state_is_local(client.trainer, client.client_id)
    assert sidecar_state["optimizer_state"]["state"]
    assert sidecar_state["scheduler_state"] is not None
    _assert_model_weight_payload(sent_payload, client.trainer.model)
