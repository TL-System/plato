"""Regressions for retained transfers and simulated-time worker reuse."""

import asyncio
import pickle
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plato.config import Config


def _server():
    from plato.servers import fedavg

    return fedavg.Server()


def _report(client_id, training_time=1, update_response=False):
    return SimpleNamespace(
        client_id=client_id,
        num_samples=1,
        accuracy=0,
        comm_time=time.time(),
        training_time=training_time,
        processing_time=0,
        update_response=update_response,
    )


def _assign(server, sid, client_id):
    server.training_clients[client_id] = {
        "id": client_id,
        "starting_round": 1,
        "start_time": 0,
        "update_requested": False,
    }
    server.training_sids.append(sid)
    server._assign_client(sid, client_id)


async def _complete(server, sid, client_id, report=None):
    await server._client_report_arrived(
        sid, client_id, pickle.dumps(report or _report(client_id))
    )
    await server._client_chunk_arrived(sid, pickle.dumps({"w": client_id}))
    await server._client_payload_arrived(sid, client_id)
    await server._client_payload_done(sid, client_id)


@pytest.mark.parametrize("minimum", [3, 4])
def test_partial_urgency_preserves_heap_order(temp_config, minimum):
    async def scenario():
        server = _server()
        server.comm_simulation = False
        server.asynchronous_mode = server.simulate_wall_time = server.request_update = (
            True
        )
        server.current_round = 20
        server.wall_time = 1
        server.staleness_bound = 10
        server.minimum_clients = minimum
        finishes = [9, 20, 58, 31, 28, 94]
        server.clients = {i: {"sid": f"s{i}", "client_id": i} for i in finishes}
        server.selected_clients = finishes[:]
        server.current_reported_clients = dict.fromkeys(finishes, True)
        server.reported_clients = [
            (
                i,
                i,
                {
                    "client_id": i,
                    "sid": f"s{i}",
                    "starting_round": 1 if i == 20 else 20,
                    "start_time": 0,
                    "report": _report(i, i),
                    "payload": {"w": i},
                },
            )
            for i in finishes
        ]
        server.sio = SimpleNamespace(emit=AsyncMock())
        server._process_reports = AsyncMock()
        server.wrap_up = AsyncMock()
        server._select_clients = AsyncMock()
        await server._process_clients(server.reported_clients[0])
        await _complete(server, "s20", 20, _report(20, 1, True))
        assert [u.client_id for u in server.updates] == [20, 9, 28, 31][:minimum]
        assert server.wall_time == {3: 28, 4: 31}[minimum]
        server._process_reports.assert_awaited_once()

    asyncio.run(scenario())


def test_earlier_response_never_rewinds_simulated_time(temp_config):
    async def scenario():
        server = _server()
        server.comm_simulation = False
        server.asynchronous_mode = server.simulate_wall_time = True
        server.current_round = 3
        server.wall_time = 50
        server.clients = {1: {"sid": "worker", "client_id": 7}}
        server.selected_clients = [7]
        server._process_reports = AsyncMock()
        server.wrap_up = AsyncMock()
        server._select_clients = AsyncMock()
        _assign(server, "worker", 7)
        await _complete(server, "worker", 7, _report(7, 2, True))
        assert server.wall_time == 50
        assert [u.client_id for u in server.updates] == [7]

    asyncio.run(scenario())


def test_batched_shared_worker_completes_every_urgent_response(temp_config):
    Config().trainer.max_concurrency = 1

    async def scenario():
        server = _server()
        server.comm_simulation = False
        server.asynchronous_mode = server.simulate_wall_time = server.request_update = (
            True
        )
        server.staleness_bound = 1
        server.minimum_clients = 1
        server.clients_per_round = 3
        server.total_clients = 10
        server.wall_time = 0
        server.clients = {100: {"sid": "worker", "client_id": 1}}
        server.choose_clients = lambda pool, count: [7, 2, 9] if count == 3 else [9]
        server.sio = SimpleNamespace(emit=AsyncMock())
        server.algorithm = SimpleNamespace(extract_weights=lambda: {"w": 0})
        aggregated = []

        async def aggregate():
            aggregated.append(
                (server.current_round, [u.client_id for u in server.updates])
            )

        server._process_reports = aggregate
        server.wrap_up = AsyncMock()
        normal = []

        async def send(sid, payload, client_id):
            normal.append((server.current_round, client_id))
            if server.current_round <= 3:
                await _complete(
                    server,
                    sid,
                    client_id,
                    _report(client_id, {7: 100, 2: 101, 9: 1}[client_id]),
                )

        server._send = send
        await server._select_clients()
        assert normal == [(1, 7), (1, 2), (1, 9), (2, 9), (3, 9)]

        def requests():
            return [
                c.args[1]["client_id"]
                for c in server.sio.emit.await_args_list
                if c.args[0] == "request_update"
            ]

        assert len(requests()) == 1
        first = requests()[0]
        assert server.training_sids == ["worker"]
        assert server._session_assignments == {"worker": first}
        # The other stale report remains retained until its worker is available.
        pending = next(info for info in server.reported_clients if info[1] in (7, 2))
        assert pending[1] != first
        before = len(aggregated)
        await _complete(server, "worker", first, _report(first, 2, True))
        assert len(aggregated) == before
        assert len(requests()) == 2 and set(requests()) == {7, 2}
        second = requests()[1]
        assert server._session_assignments == {"worker": second}
        assert server.training_sids == ["worker"]
        await _complete(server, "worker", second, _report(second, 2, True))
        assert server.current_round == 4
        assert {7, 2}.issubset(aggregated[-1][1])
        assert not any(c["update_requested"] for c in server.training_clients.values())
        assert len(server.training_sids) == len(set(server.training_sids))

    asyncio.run(asyncio.wait_for(scenario(), 2))


def test_distinct_workers_remain_concurrent_during_urgency(temp_config):
    async def scenario():
        server = _server()
        server.comm_simulation = False
        server.asynchronous_mode = server.simulate_wall_time = server.request_update = (
            True
        )
        server.current_round = 3
        server.wall_time = 1
        server.staleness_bound = 1
        server.minimum_clients = 2
        server.clients = {
            1: {"sid": "first", "client_id": 7},
            2: {"sid": "second", "client_id": 2},
        }
        server.selected_clients = [7, 2]
        server.sio = SimpleNamespace(emit=AsyncMock())
        server._process_reports = AsyncMock()
        server.wrap_up = AsyncMock()
        server._select_clients = AsyncMock()
        for sid, client_id in [("first", 7), ("second", 2)]:
            _assign(server, sid, client_id)
            await _complete(server, sid, client_id, _report(client_id, 100))
        assert server._session_assignments == {"first": 7, "second": 2}
        assert set(server.training_sids) == {"first", "second"}
        await _complete(server, "first", 7, _report(7, 2, True))
        server._process_reports.assert_not_awaited()
        await _complete(server, "second", 2, _report(2, 2, True))
        server._process_reports.assert_awaited_once()
        assert server.training_sids == []
        assert {u.client_id for u in server.updates} == {7, 2}

    asyncio.run(scenario())


def test_distinct_retained_transfers_survive_worker_reuse_and_cleanup(temp_config):
    async def scenario():
        wire_size = len(pickle.dumps(_report(7))) + len(pickle.dumps({"w": 7}))
        Config().server.max_buffered_bytes = 2 * wire_size + 20
        server = _server()
        server.comm_simulation = False
        server.asynchronous_mode = True
        server.clients_per_round = 3
        server.total_clients = 10
        server.clients = {
            1: {"sid": "first", "client_id": 1},
            2: {"sid": "second", "client_id": 7},
            3: {"sid": "third", "client_id": 3},
        }
        for sid, client_id in [("first", 1), ("second", 7)]:
            _assign(server, sid, client_id)
            await _complete(server, sid, client_id)
        _assign(server, "third", 3)
        server._release_processed_payloads()
        assert server._buffered_inbound_bytes() == 2 * wire_size
        server.choose_clients = lambda pool, count: [7, 8]
        server.sio = SimpleNamespace(emit=AsyncMock())
        server.algorithm = SimpleNamespace(extract_weights=lambda: {"w": 0})
        server._process_reports = AsyncMock()

        async def send(sid, payload, client_id):
            if sid == "first":
                await _complete(server, "first", 7)
                # Two physical workers retain distinct transfers for logical client 7.
                assert server.client_payload == {"first": {"w": 7}, "second": {"w": 7}}
                assert server._buffered_inbound_bytes() == 2 * wire_size
                with pytest.raises(ValueError, match="buffered"):
                    await server._client_report_arrived(
                        "third", 3, pickle.dumps(_report(3))
                    )
                assert server._session_assignments["third"] == 3

        server._send = send
        await server._select_clients()
        # Reassigning the second worker clears only its old completion cache.
        assert server._buffered_inbound_bytes() == wire_size
        await server._client_report_arrived("third", 3, pickle.dumps(_report(3)))
        with pytest.raises(ValueError, match="limit"):
            await server._client_chunk_arrived(
                "third", b"x" * (server.transport_limits.max_chunk_bytes + 1)
            )
        # Malformed ingress cleanup cannot release the first worker's queued update.
        assert server._buffered_inbound_bytes() == wire_size
        server._close = AsyncMock()
        await server._client_disconnected("first")
        # Its unprocessed update still owns its bytes after completion cache removal.
        assert server._buffered_inbound_bytes() == wire_size
        assert server.updates[0].payload == {"w": 7}
        server._release_processed_payloads()
        assert server._buffered_inbound_bytes() == 0
        await server._close_connections()
        assert server._buffered_inbound_bytes() == 0

    asyncio.run(scenario())
