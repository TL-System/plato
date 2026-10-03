"""Actual batched recovery preserves completed work after urgent worker loss."""

import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plato.config import Config
from tests.servers.test_runtime_review_regressions import _complete, _report, _server


async def _batched_urgent_server():
    server = _server()
    server.comm_simulation = False
    server.asynchronous_mode = server.simulate_wall_time = server.request_update = True
    server.staleness_bound = 1
    server.wall_time = 0
    server.minimum_clients = 1
    server.clients_per_round = 4
    server.total_clients = 10
    server.clients = {
        100: {"sid": "lost", "client_id": 1},
        200: {"sid": "survivor", "client_id": 2},
    }

    def choose(pool, count):
        if count == 4:
            return [7, 2, 8, 9]
        return [9 if 9 in pool else pool[0]]

    server.choose_clients = choose
    server.algorithm = SimpleNamespace(extract_weights=lambda: {"w": 0})
    server.sio = SimpleNamespace(emit=AsyncMock())
    server.wrap_up = AsyncMock()
    aggregates = []

    async def aggregate():
        aggregates.append(
            (server.current_round, server.wall_time, list(server.updates))
        )

    server._process_reports = aggregate
    normal = []

    async def send(sid, payload, client_id):
        normal.append((server.current_round, sid, client_id))
        if server.current_round <= 4:
            duration = (
                {7: 100, 2: 101, 8: 102, 9: 1}[client_id]
                if server.current_round <= 3
                else 1
            )
            await _complete(server, sid, client_id, _report(client_id, duration))

    server._send = send
    await server._select_clients()
    assert [client_id for _, _, client_id in normal] == [7, 2, 8, 9, 9, 9]
    assert server._session_assignments == {"lost": 7, "survivor": 2}
    assert server.training_sids == ["lost", "survivor"]
    return server, normal, aggregates


def _configure_recovery(debug):
    Config().trainer.max_concurrency = 2
    Config().trainer.rounds = 10
    Config().clients.total_clients = 10
    Config().clients.per_round = 4
    Config.general = (
        SimpleNamespace() if debug is None else SimpleNamespace(debug=debug)
    )


def _retained_wire_bytes(server):
    """Independent bytes of unique report/payload pairs still held by the runtime."""
    pairs = {
        id(info[2]["report"]): (info[2]["report"], info[2]["payload"])
        for info in server.reported_clients
    }
    for sid, report in server.reports.items():
        pairs[id(report)] = (report, server.client_payload[sid])
    return sum(
        len(pickle.dumps(report)) + len(pickle.dumps(payload))
        for report, payload in pairs.values()
    )


@pytest.mark.parametrize("survivor_first", [False, True])
def test_batched_urgent_loss_recovers_with_completed_reports(
    temp_config, survivor_first
):
    _configure_recovery(False)

    async def scenario():
        server, normal, aggregates = await _batched_urgent_server()
        server._close = AsyncMock()
        deferred = next(info for info in server.reported_clients if info[1] == 8)
        assert deferred[2]["sid"] == "lost"
        deferred_token = deferred[2]["transfer_id"]
        retained = _retained_wire_bytes(server)
        assert server._buffered_inbound_bytes() == retained
        raw_report = pickle.dumps(_report(7, 2, True))
        await server._client_report_arrived("lost", 7, raw_report)
        await server._client_chunk_arrived("lost", b"partial")
        assert server._buffered_inbound_bytes() == retained + len(raw_report) + 7
        try:
            if survivor_first:
                await _complete(server, "survivor", 2, _report(2, 2, True))
                assert server.current_round == 3
            await server._client_disconnected("lost")
            assert not any(peer["sid"] == "lost" for peer in server.clients.values())
            assert "lost" not in server.reports
            assert "lost" not in server._inbound_transfers
            if not survivor_first:
                # Its valid old report and bytes survive loss of the current transfer.
                assert deferred in server.reported_clients
                assert deferred_token in server._queued_payload_bytes
                assert server._buffered_inbound_bytes() == retained
                await _complete(server, "survivor", 2, _report(2, 2, True))
            state = {
                "survivor_first": survivor_first,
                "round": server.current_round,
                "assignments": dict(server._session_assignments),
                "urgent_requests": [
                    (c.kwargs["room"], c.args[1]["client_id"])
                    for c in server.sio.emit.await_args_list
                    if c.args[0] == "request_update"
                ],
                "aggregated": [
                    (r, [u.client_id for u in updates]) for r, _, updates in aggregates
                ],
            }
            print("batched disconnect recovery", state)
            # The survivor completes a full new four-client batch, reaching round 5.
            assert server.current_round == 5, state
            assert [cid for r, sid, cid in normal if r == 4 and sid == "survivor"] == [
                7,
                2,
                8,
                9,
            ]
            recovered = next(updates for r, _, updates in aggregates if r == 3)
            assert {u.client_id: u.payload for u in recovered} == {
                2: {"w": 2},
                8: {"w": 8},
                9: {"w": 9},
            }
            assert deferred_token not in server._queued_payload_bytes
            assert state["urgent_requests"] == [("lost", 7), ("survivor", 2)]
            assert set(server._session_assignments) == {"survivor"}
            assert server.training_sids == ["survivor"]
            assert not any(
                c["update_requested"] for c in server.training_clients.values()
            )
            assert server._buffered_inbound_bytes() == _retained_wire_bytes(server)
            times = [wall_time for _, wall_time, _ in aggregates]
            assert times == sorted(times)
            assert all(
                server.reported_clients[(i - 1) // 2] <= server.reported_clients[i]
                for i in range(1, len(server.reported_clients))
            )
            server._close.assert_not_awaited()
            with pytest.raises(ValueError, match="assignment"):
                await server._client_report_arrived("lost", 8, b"not a pickle")
        finally:
            await server._close_connections()
        assert server._buffered_inbound_bytes() == 0
        assert not server.reported_clients and not server.updates and not server.reports
        assert not server._session_assignments and not server._queued_payload_bytes

    asyncio.run(asyncio.wait_for(scenario(), 2))


@pytest.mark.parametrize("debug", [None, True])
def test_default_disconnect_policy_closes_batched_urgent_work(temp_config, debug):
    _configure_recovery(debug)

    async def scenario():
        server, normal, aggregates = await _batched_urgent_server()
        old_rounds = len(aggregates)
        closed = []

        async def close():
            closed.append(server.current_round)
            await server._close_connections()

        server._close = close
        await server._client_report_arrived(
            "lost", 7, pickle.dumps(_report(7, 2, True))
        )
        await server._client_chunk_arrived("lost", b"partial")
        await server._client_disconnected("lost")
        assert closed == [3]
        assert len(aggregates) == old_rounds and len(normal) == 6
        assert server._buffered_inbound_bytes() == 0
        assert not server.reported_clients and not server.updates and not server.reports
        assert not server._session_assignments and not server._queued_payload_bytes
        with pytest.raises(ValueError, match="assignment"):
            await server._client_report_arrived("survivor", 2, b"not a pickle")

    asyncio.run(asyncio.wait_for(scenario(), 2))


def test_assignment_rejects_unregistered_session_before_mutation(temp_config):
    server = _server()
    with pytest.raises(ValueError, match="unregistered"):
        server._assign_client("unknown", 7)
    assert server._session_assignments == {}
