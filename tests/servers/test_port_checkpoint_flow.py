"""PORT's actual writer and historical reader must agree on checkpoints."""

import asyncio
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from plato.config import Config
from plato.servers.strategies.aggregation.port import PortAggregationStrategy
from plato.servers.strategies.base import ServerContext
from plato.trainers.composable import ComposableTrainer


@pytest.mark.parametrize("legacy", [False, True])
def test_actual_port_hook_to_stale_similarity_and_aggregation(
    temp_config, tmp_path, legacy
):
    path = Path(__file__).resolve().parents[2] / "examples/async/port/port_server.py"
    spec = importlib.util.spec_from_file_location("phase2b_port_server", path)
    port_server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(port_server)
    Config.params["model_path"] = str(tmp_path)
    trainer = ComposableTrainer(model=lambda: torch.nn.Linear(2, 1, bias=False))
    trainer.set_client_id(0)
    model = trainer.require_model()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 1.0]]))

    server = port_server.Server()
    server.trainer = trainer
    server.current_round = 1
    # Exercise the extension hook, not a substitute writer.
    server.weights_aggregated([])
    assert (tmp_path / "model_1.safetensors").exists()
    assert (tmp_path / "model_1.safetensors.pkl").exists()
    if legacy:
        torch.save(model.state_dict(), tmp_path / "model_1.pth")
        (tmp_path / "model_1.safetensors").unlink()

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 2.0]]))
    model.train()
    context = ServerContext()
    context.trainer = trainer
    context.current_round = 3
    strategy = PortAggregationStrategy()
    delta = {"weight": torch.tensor([[2.0, -1.0]])}
    similarity = asyncio.run(strategy._cosine_similarity(delta, 2, context))
    # Independent dot product of [2,1] with [2,-1], divided by norms.
    expected_similarity = 3 / (math.sqrt(5) * math.sqrt(5))
    assert similarity == pytest.approx(expected_similarity)

    updates = [
        SimpleNamespace(report=SimpleNamespace(num_samples=2), staleness=2),
        SimpleNamespace(report=SimpleNamespace(num_samples=6), staleness=0),
    ]
    second_delta = {"weight": torch.tensor([[-1.0, 4.0]])}
    result = asyncio.run(
        strategy.aggregate_deltas(updates, [delta, second_delta], context)
    )
    stale_weight = 0.25 * ((expected_similarity + 1) / 2 + 10 / 12)
    fresh_weight = 0.75 * (1 + 1)
    ratio = stale_weight / (stale_weight + fresh_weight)
    expected = torch.tensor([[2 * ratio - (1 - ratio), -ratio + 4 * (1 - ratio)]])
    assert torch.allclose(result["weight"], expected)
    assert torch.equal(model.weight, torch.tensor([[3.0, 2.0]]))
    assert model.training
