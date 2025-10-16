"""Tests for the FedAvg GAN aggregation strategy."""

import asyncio
from types import SimpleNamespace

import torch

from plato.servers.strategies.aggregation import FedAvgGanAggregationStrategy
from plato.servers.strategies.base import ServerContext


class DummyTrainer:
    def __init__(self):
        self.device = torch.device("cpu")

    def zeros(self, shape):
        return torch.zeros(shape, device=self.device)


class DummyServer:
    def __init__(self):
        self.total_samples = 0


def _build_context():
    context = ServerContext()
    context.trainer = DummyTrainer()
    context.algorithm = SimpleNamespace()
    context.server = DummyServer()
    return context


def test_gan_aggregation_strategy_weighted_average():
    context = _build_context()
    strategy = FedAvgGanAggregationStrategy()
    strategy.setup(context)

    gen_update_a = {"w": torch.tensor([1.0, 2.0])}
    disc_update_a = {"dw": torch.tensor([3.0])}

    gen_update_b = {"w": torch.tensor([3.0, 4.0])}
    disc_update_b = {"dw": torch.tensor([9.0])}

    deltas_received = [
        (gen_update_a, disc_update_a),
        (gen_update_b, disc_update_b),
    ]

    updates = [
        SimpleNamespace(report=SimpleNamespace(num_samples=2)),
        SimpleNamespace(report=SimpleNamespace(num_samples=1)),
    ]

    aggregated_gen, aggregated_disc = asyncio.run(
        strategy.aggregate_deltas(updates, deltas_received, context)
    )

    expected_gen = torch.tensor([5.0 / 3.0, 8.0 / 3.0])
    expected_disc = torch.tensor([15.0 / 3.0])

    assert torch.allclose(aggregated_gen["w"], expected_gen)
    assert torch.allclose(aggregated_disc["dw"], expected_disc)
