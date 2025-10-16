"""Tests for the Hermes aggregation strategy."""

import asyncio
from types import SimpleNamespace

import numpy as np
import torch

from plato.servers.strategies.aggregation.hermes import HermesAggregationStrategy
from plato.servers.strategies.base import ServerContext


class DummyTrainer:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.device = torch.device("cpu")

    def zeros(self, shape):
        return torch.zeros(shape, device=self.device)


class DummyAlgorithm:
    def __init__(self, baseline):
        self.current = {name: tensor.clone() for name, tensor in baseline.items()}

    def compute_weight_deltas(self, baseline_weights, weights_list):
        return [
            {name: weights[name] - baseline_weights[name] for name in weights.keys()}
            for weights in weights_list
        ]

    def update_weights(self, deltas):
        for name, delta in deltas.items():
            self.current[name] = self.current[name] + delta
        return {name: tensor.clone() for name, tensor in self.current.items()}


class DummyServer:
    def __init__(self):
        self.total_samples = 0
        self.masks_received = []
        self.aggregated_clients_model = {}

    def update_client_model(self, aggregated, updates):
        for weights, update in zip(aggregated, updates):
            self.aggregated_clients_model[update.client_id] = weights


def _build_context(baseline):
    context = ServerContext()
    context.trainer = DummyTrainer()
    context.algorithm = DummyAlgorithm(baseline)
    context.server = DummyServer()
    context.state["prng_state"] = None
    return context


def test_hermes_aggregation_strategy_produces_weighted_average():
    baseline = {
        "linear.weight": torch.zeros(2, 2),
        "linear.bias": torch.zeros(2),
    }

    context = _build_context(baseline)

    strategy = HermesAggregationStrategy()
    strategy.setup(context)

    context.server.masks_received = [
        [np.ones((2, 2), dtype=int)],
        [np.ones((2, 2), dtype=int)],
    ]

    weights_received = [
        {
            "linear.weight": torch.ones(2, 2),
            "linear.bias": torch.ones(2),
        },
        {
            "linear.weight": torch.full((2, 2), 2.0),
            "linear.bias": torch.full((2,), 2.0),
        },
    ]

    updates = [
        SimpleNamespace(
            client_id=0,
            report=SimpleNamespace(num_samples=2),
        ),
        SimpleNamespace(
            client_id=1,
            report=SimpleNamespace(num_samples=1),
        ),
    ]

    context.updates = updates

    aggregated = asyncio.run(
        strategy.aggregate_weights(updates, baseline, weights_received, context)
    )

    expected_weight = torch.full((2, 2), 4.0 / 3.0)
    expected_bias = torch.full((2,), 4.0 / 3.0)

    assert torch.allclose(aggregated["linear.weight"], expected_weight, atol=1e-6)
    assert torch.allclose(aggregated["linear.bias"], expected_bias, atol=1e-6)

    assert set(context.server.aggregated_clients_model.keys()) == {0, 1}
