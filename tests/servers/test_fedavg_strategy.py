"""Tests for FedAvg aggregation and algorithm utilities."""

import asyncio
from types import SimpleNamespace

import torch

from plato.servers.strategies.aggregation import FedAvgAggregationStrategy
from plato.servers.strategies.base import ServerContext
from plato.trainers.composable import ComposableTrainer


def test_fedavg_aggregation_weighted_mean(temp_config):
    """FedAvg aggregation should compute the weighted mean of client deltas."""
    trainer = ComposableTrainer(model=lambda: torch.nn.Linear(2, 1))
    trainer.set_client_id(0)

    context = ServerContext()
    context.trainer = trainer

    deltas = [
        {"weight": torch.ones((1, 2)), "bias": torch.tensor([0.5])},
        {"weight": torch.full((1, 2), 3.0), "bias": torch.tensor([1.5])},
    ]
    updates = [
        SimpleNamespace(report=SimpleNamespace(num_samples=10)),
        SimpleNamespace(report=SimpleNamespace(num_samples=30)),
    ]

    aggregated = asyncio.run(
        FedAvgAggregationStrategy().aggregate_deltas(updates, deltas, context)
    )

    expected_weight = deltas[0]["weight"] * 0.25 + deltas[1]["weight"] * 0.75
    expected_bias = deltas[0]["bias"] * 0.25 + deltas[1]["bias"] * 0.75

    assert torch.allclose(aggregated["weight"], expected_weight)
    assert torch.allclose(aggregated["bias"], expected_bias)
