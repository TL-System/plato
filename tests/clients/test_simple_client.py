"""End-to-end smoke tests for the strategy-based client runtime."""

import asyncio
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from plato.algorithms import fedavg
from plato.clients import simple
from plato.config import Config
from plato.trainers.composable import ComposableTrainer


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


def _build_client():
    """Instantiate a client wired with custom model, datasource, and trainer."""
    return simple.Client(
        model=torch.nn.Linear(4, 2),
        datasource=ToyDatasource,
        trainer=ComposableTrainer,
        algorithm=lambda trainer: fedavg.Algorithm(trainer),
    )


def test_simple_client_trains_with_default_strategies(temp_config):
    """A simple client should complete one training round using the strategy stack."""
    Config().trainer = Config().trainer._replace(epochs=1, batch_size=2)

    client = _build_client()

    # Assign identifiers expected by the client runtime.
    client.client_id = 1
    client._context.client_id = 1
    client.current_round = 1
    client._context.current_round = 1

    # Prepare data and runtime components.
    client._load_data()
    client.configure()
    client._allocate_data()

    report, payload = asyncio.run(client._train())

    assert report.client_id == 1
    # With partition_size=4 each client receives four samples.
    assert report.num_samples == 4
    assert isinstance(payload, dict)
    assert all(isinstance(value, torch.Tensor) for value in payload.values())
