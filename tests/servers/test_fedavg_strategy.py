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


def test_fedavg_aggregation_skips_feature_payloads(temp_config):
    """Feature updates should be ignored by the FedAvg aggregator."""
    trainer = ComposableTrainer(model=lambda: torch.nn.Linear(2, 1))
    trainer.set_client_id(0)

    context = ServerContext()
    context.trainer = trainer

    updates = [
        SimpleNamespace(report=SimpleNamespace(num_samples=10, type="features")),
    ]

    model = trainer.model
    assert model is not None
    aggregated = asyncio.run(
        FedAvgAggregationStrategy().aggregate_weights(
            updates, model.state_dict(), [{}], context
        )
    )

    assert aggregated is None


class DummyAlgorithm:
    """Minimal algorithm stub for server aggregation dispatch tests."""

    def __init__(self, baseline):
        self.current = {name: tensor.clone() for name, tensor in baseline.items()}

    def extract_weights(self):
        return {name: tensor.clone() for name, tensor in self.current.items()}

    def compute_weight_deltas(self, baseline_weights, weights_list):
        return [
            {
                name: weights[name] - baseline_weights[name]
                for name in baseline_weights.keys()
            }
            for weights in weights_list
        ]

    def update_weights(self, deltas):
        self.current = {
            name: self.current[name] + deltas[name] for name in self.current.keys()
        }
        return self.extract_weights()

    def load_weights(self, weights):
        self.current = {name: tensor.clone() for name, tensor in weights.items()}


class DeltaOnlyStrategy(FedAvgAggregationStrategy):
    """Strategy overriding only delta aggregation to exercise dispatch."""

    def __init__(self):
        super().__init__()
        self.delta_calls = 0

    async def aggregate_deltas(self, updates, deltas_received, context):
        self.delta_calls += 1
        return await super().aggregate_deltas(updates, deltas_received, context)


def test_fedavg_server_prefers_custom_delta_strategy_over_inherited_weights(
    temp_config,
):
    """Custom delta strategies should not be bypassed by inherited weight hooks."""
    from plato.config import Config
    from plato.servers import fedavg

    Config().server.do_test = False

    strategy = DeltaOnlyStrategy()
    server = fedavg.Server(aggregation_strategy=strategy)

    baseline = {"weight": torch.zeros((1, 2)), "bias": torch.zeros(1)}
    server.algorithm = DummyAlgorithm(baseline)
    server.context.algorithm = server.algorithm
    server.context.server = server
    server.context.state["prng_state"] = None

    server.updates = [
        SimpleNamespace(
            client_id=1,
            report=SimpleNamespace(
                num_samples=1,
                accuracy=0.5,
                processing_time=0.1,
                comm_time=0.1,
                training_time=0.1,
            ),
            payload={
                "weight": torch.ones((1, 2)),
                "bias": torch.ones(1),
            },
        )
    ]

    asyncio.run(server._process_reports())

    assert strategy.delta_calls == 1
    assert torch.allclose(server.algorithm.current["weight"], torch.ones((1, 2)))
    assert torch.allclose(server.algorithm.current["bias"], torch.ones(1))
