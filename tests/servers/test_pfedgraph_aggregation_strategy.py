"""Tests for pFedGraph aggregation strategy."""

from types import SimpleNamespace

import asyncio
import torch

from plato.servers.strategies.aggregation.pfedgraph import (
    PFedGraphAggregationStrategy,
    _project_to_simplex,
)
from plato.servers.strategies.base import ServerContext


class DummyServer:
    """Minimal server stub to capture client models."""

    def __init__(self, total_clients: int):
        self.total_clients = total_clients
        self.client_models: dict[int, dict[str, torch.Tensor]] = {}
        self.total_samples = 0

    def update_client_model(self, aggregated_clients_models, updates):
        for model, update in zip(aggregated_clients_models, updates):
            self.client_models[update.client_id] = model


def test_project_to_simplex_returns_probability_vector():
    vector = torch.tensor([0.2, 0.2, 0.2])
    projected = _project_to_simplex(vector)
    assert torch.all(projected >= 0)
    assert torch.isclose(projected.sum(), torch.tensor(1.0), atol=1e-6)


def test_pfedgraph_aggregation_updates_client_models():
    strategy = PFedGraphAggregationStrategy(alpha=0.5, similarity_metric="all")
    context = ServerContext()
    context.server = DummyServer(total_clients=2)

    baseline_weights = {"w": torch.tensor([0.5])}
    weights_received = [
        {"w": torch.tensor([1.0])},
        {"w": torch.tensor([2.0])},
    ]

    updates = [
        SimpleNamespace(client_id=0, report=SimpleNamespace(num_samples=1)),
        SimpleNamespace(client_id=1, report=SimpleNamespace(num_samples=1)),
    ]

    strategy.setup(context)
    global_weights = asyncio.run(
        strategy.aggregate_weights(updates, baseline_weights, weights_received, context)
    )

    assert context.server.client_models
    for client_weights in context.server.client_models.values():
        assert torch.isclose(client_weights["w"], torch.tensor([1.5]), atol=1e-6)

    assert global_weights is not None
    assert torch.isclose(global_weights["w"], torch.tensor([1.5]), atol=1e-6)
