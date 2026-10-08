"""Lifecycle contracts beyond ordinary synchronous FedAvg dispatch."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from plato.config import Config
from plato.servers.strategies.aggregation import (
    FedAsyncAggregationStrategy,
    FedBuffAggregationStrategy,
)
from plato.servers.strategies.base import ServerContext


def test_fedbuff_preserves_uniform_research_weighting(temp_config):
    context = ServerContext()
    context.trainer = SimpleNamespace(zeros=torch.zeros)
    reports = [
        SimpleNamespace(report=SimpleNamespace(num_samples=count)) for count in (2, 6)
    ]
    deltas = [{"w": torch.tensor([value])} for value in (1.0, 5.0)]
    result = asyncio.run(
        FedBuffAggregationStrategy().aggregate_deltas(reports, deltas, context)
    )
    assert result["w"].item() == pytest.approx(3.0)


def test_fedasync_actual_example_preserves_staleness_mixing(temp_config):
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/async/fedasync/fedasync_algorithm.py"
    )
    spec = importlib.util.spec_from_file_location("phase2b_fedasync_algorithm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = ServerContext()
    context.algorithm = module.Algorithm(trainer=None)
    strategy = FedAsyncAggregationStrategy(
        mixing_hyperparameter=0.9,
        adaptive_mixing=True,
        staleness_func_type="polynomial",
        staleness_func_params={"a": 1},
    )
    updates = [SimpleNamespace(report=SimpleNamespace(num_samples=2), staleness=2)]
    result = asyncio.run(
        strategy.aggregate_weights(
            updates, {"w": torch.tensor([10.0])}, [{"w": torch.tensor([2.0])}], context
        )
    )
    # Staleness 2 reduces mixing to 0.9 / 3, giving 0.7*10 + 0.3*2.
    assert result["w"].item() == pytest.approx(7.6)


def test_async_periodic_aggregation_respects_minimum_and_round_order(temp_config):
    from plato.servers import fedavg

    async def scenario():
        server = fedavg.Server()
        server.asynchronous_mode = True
        server.minimum_clients = 2
        events = []

        async def aggregate():
            events.append("aggregate")

        async def wrap():
            events.append("wrap")

        async def select():
            events.append("select")

        setattr(server, "_process_reports", aggregate)
        setattr(server, "wrap_up", wrap)
        setattr(server, "_select_clients", select)
        server.updates = [SimpleNamespace(client_id=1)]
        await server._periodic_task()
        assert events == []
        server.updates.append(SimpleNamespace(client_id=2))
        await server._periodic_task()
        assert events == ["aggregate", "wrap", "select"]

    asyncio.run(scenario())


def test_cross_silo_invalid_hook_output_preserves_trainer_identity(temp_config):
    from plato.servers import fedavg_cs

    server = fedavg_cs.Server()
    server.trainer = SimpleNamespace(client_id=9)

    def set_client_id(client_id):
        assert server.trainer is not None
        server.trainer.client_id = client_id

    server.trainer.set_client_id = set_client_id
    server.updates = [
        SimpleNamespace(report=SimpleNamespace(num_samples=1), payload={})
    ]
    setattr(server, "weights_received", lambda weights: [])
    with pytest.raises(ValueError, match="payload"):
        asyncio.run(server._process_reports())
    assert server.trainer.client_id == 9


@pytest.mark.parametrize("batched", [False, True])
def test_async_partial_round_replaces_only_idle_workers(temp_config, batched):
    from unittest.mock import AsyncMock

    from plato.servers import fedavg

    if batched:
        Config().trainer.max_concurrency = 1

    async def scenario():
        server = fedavg.Server()
        server.comm_simulation = False
        server.asynchronous_mode = True
        server.minimum_clients = 1
        server.total_clients = 4
        server.clients_per_round = 3 if batched else 2
        server.clients = {
            100: {"sid": "slow", "client_id": 1},
            200: {"sid": "idle", "client_id": 2},
        }
        server.current_round = 1
        server.selected_clients = [1, 2]
        server.training_clients = {
            1: {
                "id": 1,
                "starting_round": 1,
                "start_time": 0,
                "update_requested": False,
            }
        }
        server.training_sids = ["slow"]
        server._assign_client("slow", 1)
        server.updates = [SimpleNamespace(client_id=2)]
        setattr(server, "sio", SimpleNamespace(emit=AsyncMock()))
        server.algorithm = SimpleNamespace(extract_weights=lambda: {})
        server._send = AsyncMock()
        server._process_reports = AsyncMock()
        server.wrap_up = AsyncMock()
        await server._periodic_task()
        assert server._session_assignments["slow"] == 1
        assert server.training_clients[1]["starting_round"] == 1
        assert 1 not in server.selected_clients
        assert len(server.selected_clients) == server.clients_per_round - 1
        assert server.training_sids == ["slow", "idle"]
        assert server._send.await_count == 1
        assert server._send.call_args.args[0] == "idle"

    asyncio.run(scenario())
