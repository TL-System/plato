"""Exercise actual ingress handlers with inert trusted-peer payloads."""

import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def assigned_server():
    from plato.servers import fedavg

    server = fedavg.Server()
    server.comm_simulation = False
    server.clients = {
        100: {"sid": "worker", "client_id": 7},
        200: {"sid": "other", "client_id": 2},
    }
    server.training_clients = {
        7: {"id": 7, "starting_round": 1, "start_time": 0},
        2: {"id": 2, "starting_round": 1, "start_time": 0},
    }
    server.training_sids = ["worker", "other"]
    if hasattr(server, "_assign_client"):
        server._assign_client("worker", 7)
        server._assign_client("other", 2)
    return server


def report_bytes(client_id):
    return pickle.dumps(
        SimpleNamespace(
            client_id=client_id,
            num_samples=1,
            comm_time=0,
            training_time=1,
            processing_time=0,
            update_response=False,
        )
    )


@pytest.mark.parametrize("sid,claimed", [("worker", 2), ("unknown", 2)])
def test_report_rejects_unassigned_identity_before_parsing(temp_config, sid, claimed):
    server = assigned_server()
    server.reports["other"] = "untouched"
    with pytest.raises(ValueError, match="assign|session"):
        asyncio.run(server._client_report_arrived(sid, claimed, b"not-pickle"))
    assert server.reports == {"other": "untouched"}
    assert server.client_chunks == {}


def test_report_embedded_identity_cannot_target_another_client(temp_config):
    server = assigned_server()
    with pytest.raises(ValueError, match="identity|client"):
        asyncio.run(server._client_report_arrived("worker", 7, report_bytes(2)))
    assert server.reports == {}
    assert server.client_chunks == {}


@pytest.mark.parametrize("sid", ["unknown", "stale"])
@pytest.mark.parametrize("stage", ["chunk", "part", "done", "s3"])
def test_unknown_or_stale_session_rejected_at_every_payload_stage(
    temp_config, sid, stage
):
    async def scenario():
        server = assigned_server()
        await server._client_report_arrived("other", 2, report_bytes(2))
        if sid == "stale":
            await server._client_report_arrived("worker", 7, report_bytes(7))
            del server.clients[100]
            session = "worker"
        else:
            session = sid
        server.s3_client = SimpleNamespace(
            receive_from_s3=lambda key: pytest.fail("S3 lookup")
        )
        with pytest.raises(ValueError, match="session|Session|assignment"):
            if stage == "chunk":
                await server._client_chunk_arrived(session, b"not-pickle")
            elif stage == "part":
                await server._client_payload_arrived(session, 2)
            else:
                await server._client_payload_done(
                    session,
                    2,
                    s3_key="client_payload_2_ABC123" if stage == "s3" else None,
                )
        assert server.reports["other"].client_id == 2
        assert server.client_chunks["other"] == []

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["part", "done", "s3"])
def test_payload_identity_checked_before_parsing_or_s3(temp_config, stage):
    server = assigned_server()
    asyncio.run(server._client_report_arrived("worker", 7, report_bytes(7)))
    asyncio.run(server._client_chunk_arrived("worker", pickle.dumps({"w": 1})))
    server.s3_client = SimpleNamespace(receive_from_s3=lambda key: pytest.fail(key))
    server.process_client_info = AsyncMock()
    operation = (
        server._client_payload_arrived("worker", 2)
        if stage == "part"
        else server._client_payload_done(
            "worker", 2, s3_key="client_payload_2_ABC123" if stage == "s3" else None
        )
    )
    with pytest.raises(ValueError, match="assign|session"):
        asyncio.run(operation)
    server.process_client_info.assert_not_awaited()
    assert 2 in server.training_clients


@pytest.mark.parametrize(
    "bad", [b"broken", pickle.dumps(1)[:-1], pickle.dumps(1) + b"x"]
)
def test_malformed_part_releases_only_sender_buffers(temp_config, bad):
    async def scenario():
        server = assigned_server()
        await server._client_report_arrived("worker", 7, report_bytes(7))
        await server._client_report_arrived("other", 2, report_bytes(2))
        await server._client_chunk_arrived("worker", bad)
        with pytest.raises((ValueError, pickle.UnpicklingError, EOFError)):
            await server._client_payload_arrived("worker", 7)
        assert "worker" not in server.reports
        assert "worker" not in server.client_chunks
        assert "other" in server.reports
        await server._client_chunk_arrived("other", pickle.dumps({"w": 2}))
        await server._client_payload_arrived("other", 2)
        assert server.client_payload["other"] == {"w": 2}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "limit", ["chunk_bytes", "payload_bytes", "chunks", "parts", "buffered"]
)
def test_server_limits_apply_across_parts_and_preserve_other_client(temp_config, limit):
    from plato.config import Config

    raw = pickle.dumps({"w": 1})
    if limit == "chunk_bytes":
        Config().server.max_chunk_bytes = len(raw) - 1
    elif limit == "payload_bytes":
        Config().server.max_payload_bytes = len(raw)
    elif limit == "chunks":
        Config().server.max_payload_chunks = 1
    elif limit == "parts":
        Config().server.max_payload_parts = 1
    else:
        Config().server.max_buffered_bytes = len(report_bytes(7)) * 2 + len(raw)

    async def scenario():
        server = assigned_server()
        await server._client_report_arrived("worker", 7, report_bytes(7))
        await server._client_report_arrived("other", 2, report_bytes(2))
        if limit != "chunk_bytes":
            await server._client_chunk_arrived("worker", raw)
            await server._client_payload_arrived("worker", 7)
        with pytest.raises(ValueError, match="limit|count"):
            await server._client_chunk_arrived("worker", raw)
            await server._client_payload_arrived("worker", 7)
        assert "worker" not in server.client_chunks
        assert "worker" not in server.reports
        assert "other" in server.reports

    asyncio.run(scenario())


def test_report_limit_rejects_before_pickle_parsing(temp_config, monkeypatch):
    from plato.config import Config
    from plato.servers import base

    Config().server.max_report_bytes = 16
    server = assigned_server()
    monkeypatch.setattr(
        base, "load_pickle", lambda data: pytest.fail("parsed oversized report")
    )
    with pytest.raises(ValueError, match="report.*limit"):
        asyncio.run(server._client_report_arrived("worker", 7, b"x" * 17))
    assert server.reports == {}


@pytest.mark.parametrize("with_part", [False, True])
def test_incomplete_transfer_expires_without_next_message(temp_config, with_part):
    from plato.config import Config

    Config().server.payload_timeout = 0.03

    async def scenario():
        server = assigned_server()
        await server._client_report_arrived("worker", 7, report_bytes(7))
        await server._client_chunk_arrived("worker", pickle.dumps({"w": 1}))
        if with_part:
            await server._client_payload_arrived("worker", 7)
        await asyncio.sleep(0.06)
        assert server.reports == {}
        assert server.client_chunks == {}
        assert server.client_payload == {}
        with pytest.raises(ValueError, match="active"):
            await server._client_payload_done("worker", 7)
        await server._client_report_arrived("other", 2, report_bytes(2))
        assert server.reports["other"].client_id == 2

    asyncio.run(scenario())


def test_duplicate_report_does_not_replace_active_payload(temp_config):
    async def scenario():
        server = assigned_server()
        await server._client_report_arrived("worker", 7, report_bytes(7))
        raw = pickle.dumps({"w": 1})
        await server._client_chunk_arrived("worker", raw)
        with pytest.raises(ValueError, match="incomplete"):
            await server._client_report_arrived("worker", 7, report_bytes(7))
        assert server.client_chunks["worker"] == [raw]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "key", ["client_payload_2_ABC123", "client_payload_7_ABC123/extra", "other"]
)
def test_s3_reference_cannot_borrow_another_client_key(temp_config, key):
    async def scenario():
        server = assigned_server()
        await server._client_report_arrived("worker", 7, report_bytes(7))
        server.s3_client = SimpleNamespace(
            receive_from_s3=lambda key: pytest.fail("S3 lookup")
        )
        with pytest.raises(ValueError, match="key.*assignment"):
            await server._client_payload_done("worker", 7, s3_key=key)
        assert "worker" not in server.reports

    asyncio.run(scenario())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_socket_completion_round_boundaries_and_cleanup(temp_config, asynchronous):
    async def scenario():
        server = assigned_server()
        server.asynchronous_mode = asynchronous
        server._process_reports = AsyncMock()
        server.wrap_up = AsyncMock()
        server._select_clients = AsyncMock()
        for sid, client_id in [("worker", 7), ("other", 2)]:
            await server._client_report_arrived(sid, client_id, report_bytes(client_id))
            await server._client_chunk_arrived(sid, pickle.dumps({"w": client_id}))
            await server._client_payload_arrived(sid, client_id)
            await server._client_payload_done(sid, client_id)
            if sid == "worker":
                server._process_reports.assert_not_awaited()
        assert [u.client_id for u in server.updates] == [7, 2]
        assert server.training_clients == {}
        assert server.training_sids == []
        assert server._inbound_transfers == {}
        assert all(chunks == [] for chunks in server.client_chunks.values())
        assert server.reports["worker"].client_id == 7
        assert server.client_payload["worker"] == {"w": 7}
        server._process_reports.assert_awaited_once()
        server.wrap_up.assert_awaited_once()
        server._select_clients.assert_awaited_once()
        with pytest.raises(ValueError, match="assignment"):
            await server._client_report_arrived("worker", 7, report_bytes(7))

    asyncio.run(scenario())


def test_reused_worker_accepts_only_new_server_issued_assignment(temp_config):
    async def scenario():
        server = assigned_server()
        server.clients_per_round = 1
        server.clients = {100: {"sid": "worker", "client_id": 1}}
        server.training_sids = []
        server.training_clients = {}
        server.algorithm = SimpleNamespace(extract_weights=lambda: {"w": 0})
        server.sio = SimpleNamespace(emit=AsyncMock())
        server._send = AsyncMock()
        server._process_clients = AsyncMock()
        for client_id in [7, 8]:
            server.choose_clients = lambda pool, count: [client_id]
            await server._select_clients()
            assert server.clients[100]["client_id"] == client_id
            if client_id == 8:
                with pytest.raises(ValueError, match="assignment"):
                    await server._client_report_arrived("worker", 7, report_bytes(7))
            await server._client_report_arrived(
                "worker", client_id, report_bytes(client_id)
            )
            await server._client_chunk_arrived("worker", pickle.dumps({"w": client_id}))
            await server._client_payload_arrived("worker", client_id)
            await server._client_payload_done("worker", client_id)
        assert server._process_clients.await_count == 2

    asyncio.run(scenario())


def test_actual_urgent_request_and_writer_use_requested_logical_client(temp_config):
    from plato.clients.strategies.base import ClientContext
    from plato.clients.strategies.defaults import DefaultCommunicationStrategy

    async def scenario():
        server = assigned_server()
        server.clients[100]["client_id"] = 9
        server._session_assignments.clear()
        server.training_clients = {}
        server.training_sids = []
        server.selected_clients = [7]
        server.current_reported_clients = {7: True}
        server.current_round = 3
        server.request_update = True
        server.asynchronous_mode = True
        server.simulate_wall_time = True
        info = (
            1,
            7,
            {
                "client_id": 7,
                "starting_round": 1,
                "start_time": 0,
                "sid": "worker",
                "report": pickle.loads(report_bytes(7)),
                "payload": {"w": 1},
            },
        )
        server.reported_clients = [info]
        server.should_request_update = lambda **kwargs: True
        server.sio = SimpleNamespace(emit=AsyncMock())
        await server._process_clients(info)
        server.sio.emit.assert_awaited_once_with(
            "request_update", {"client_id": 7, "time": server.wall_time}, room="worker"
        )
        server._process_clients = AsyncMock()
        events = []

        async def emit(event, data):
            events.append(event)
            if event == "client_report":
                await server._client_report_arrived(
                    "worker", data["id"], data["report"]
                )
            elif event == "chunk":
                await server._client_chunk_arrived("worker", data["data"])
            elif event == "client_payload":
                await server._client_payload_arrived("worker", data["id"])
            elif event == "client_payload_done":
                await server._client_payload_done("worker", data["id"])

        context = ClientContext()
        context.client_id = 9
        context.comm_simulation = False
        context.sio = SimpleNamespace(emit=emit)
        report = pickle.loads(report_bytes(7))
        report.update_response = True
        await DefaultCommunicationStrategy().send_report_and_payload(
            context, report, {"w": 7}
        )
        assert context.client_id == 9
        assert "outbound_client_id" not in context.state
        assert events == [
            "client_report",
            "chunk",
            "client_payload",
            "client_payload_done",
        ]
        client_info = server._process_clients.call_args.args[0]
        assert client_info[2]["client_id"] == 7
        assert client_info[2]["payload"] == {"w": 7}

    asyncio.run(scenario())


def test_all_buffered_stale_clients_receive_urgent_requests(temp_config):
    async def scenario():
        server = assigned_server()
        server.training_clients = {}
        server.training_sids = []
        server.selected_clients = [7, 2]
        server.current_reported_clients = {7: True, 2: True}
        server.current_round = 3
        server.request_update = server.asynchronous_mode = server.simulate_wall_time = (
            True
        )
        server.reported_clients = [
            (
                1,
                client_id,
                {
                    "client_id": client_id,
                    "starting_round": 1,
                    "start_time": 0,
                    "sid": sid,
                    "report": pickle.loads(report_bytes(client_id)),
                },
            )
            for sid, client_id in [("worker", 7), ("other", 2)]
        ]
        server.should_request_update = lambda **kwargs: True
        server.sio = SimpleNamespace(emit=AsyncMock())
        await server._process_clients(server.reported_clients[0])
        requested = [
            call.args[1]["client_id"] for call in server.sio.emit.await_args_list
        ]
        assert requested == [7, 2]
        assert server.reported_clients == []

    asyncio.run(scenario())


def test_disconnect_cleans_urgent_assignment_without_touching_other_client(temp_config):
    async def scenario():
        server = assigned_server()
        server.clients[100]["client_id"] = 9
        server.training_clients[9] = {"id": 9}
        server._close = AsyncMock()
        await server._client_report_arrived("worker", 7, report_bytes(7))
        await server._client_disconnected("worker")
        assert set(server.training_clients) == {2, 9}
        assert server.training_sids == ["other"]
        assert server.reports == server.client_chunks == server.client_payload == {}
        with pytest.raises(ValueError, match="assignment"):
            await server._client_report_arrived("worker", 7, report_bytes(7))
        await server._client_report_arrived("other", 2, report_bytes(2))

    asyncio.run(scenario())


def test_cross_silo_edge_identity_follows_actual_central_selection(
    temp_config, monkeypatch
):
    from plato.config import Config

    monkeypatch.setattr(Config, "is_central_server", staticmethod(lambda: True))

    async def scenario():
        server = assigned_server()
        server.clients = {100: {"sid": "edge", "client_id": 1}}
        server.clients_per_round = 1
        server.training_clients = {}
        server.training_sids = []
        server.algorithm = SimpleNamespace(extract_weights=lambda: {"w": 0})
        server.sio = SimpleNamespace(emit=AsyncMock())
        server._send = AsyncMock()
        server._process_clients = AsyncMock()
        await server._select_clients()
        report = pickle.loads(report_bytes(100))
        report.edge_server_comm_overhead = 7
        await server._client_report_arrived("edge", 100, pickle.dumps(report))
        await server._client_chunk_arrived("edge", pickle.dumps({"w": 1}))
        await server._client_payload_arrived("edge", 100)
        await server._client_payload_done("edge", 100)
        assert server._process_clients.call_args.args[0][2]["client_id"] == 100
        assert server.comm_overhead >= 7

    asyncio.run(scenario())


def test_completed_unaggregated_payloads_still_count_toward_buffer_limit(temp_config):
    from plato.config import Config

    raw = pickle.dumps({"w": 7})
    Config().server.max_buffered_bytes = len(report_bytes(7)) + len(raw)

    async def scenario():
        server = assigned_server()
        server._process_clients = AsyncMock()
        await server._client_report_arrived("worker", 7, report_bytes(7))
        await server._client_chunk_arrived("worker", raw)
        await server._client_payload_arrived("worker", 7)
        await server._client_payload_done("worker", 7)
        with pytest.raises(ValueError, match="buffered"):
            await server._client_report_arrived("other", 2, report_bytes(2))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kind", ["valid", "oversized", "streamed_oversize", "truncated", "incomplete"]
)
def test_s3_ingress_uses_bounded_actual_http_reader(temp_config, kind):
    from aiohttp import web

    from plato.config import Config

    Config().server.max_payload_bytes = 32
    Config().server.payload_timeout = 0.1

    async def scenario():
        async def handler(request):
            if kind == "valid":
                return web.Response(body=pickle.dumps({"w": 7}))
            if kind == "truncated":
                return web.Response(body=pickle.dumps({"w": 7})[:-1])
            if kind == "oversized":
                return web.Response(body=b"x" * 33)
            response = web.StreamResponse()
            await response.prepare(request)
            if kind == "streamed_oversize":
                await response.write(b"x" * 33)
            else:
                await response.write(b"x")
                await asyncio.sleep(0.2)
            return response

        app = web.Application()
        app.router.add_get("/object", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert isinstance(site._server, asyncio.Server)
        port = site._server.sockets[0].getsockname()[1]
        seen = []

        def presign(**kwargs):
            seen.append(kwargs)
            return f"http://127.0.0.1:{port}/object"

        try:
            server = assigned_server()
            server.s3_client = SimpleNamespace(
                key_prefix="prefix",
                bucket="bucket",
                s3_client=SimpleNamespace(generate_presigned_url=presign),
            )
            server._process_clients = AsyncMock()
            await server._client_report_arrived("worker", 7, report_bytes(7))
            if kind == "valid":
                await server._client_payload_done(
                    "worker", 7, s3_key="client_payload_7_ABC123"
                )
                assert server._process_clients.call_args.args[0][2]["payload"] == {
                    "w": 7
                }
            else:
                with pytest.raises(
                    (ValueError, pickle.UnpicklingError, EOFError, TimeoutError)
                ):
                    await server._client_payload_done(
                        "worker", 7, s3_key="client_payload_7_ABC123"
                    )
                server._process_clients.assert_not_awaited()
            assert server._inbound_transfers == {}
            if kind == "valid":
                assert server.client_payload["worker"] == {"w": 7}
                assert server.client_chunks["worker"] == []
            else:
                assert (
                    server.reports
                    == server.client_chunks
                    == server.client_payload
                    == {}
                )
            assert seen[0]["Params"]["Key"] == "prefix/client_payload_7_ABC123"
        finally:
            await runner.cleanup()

    asyncio.run(scenario())
