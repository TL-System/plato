"""Bounded object ingress preserves exactly-once prefix and byte contracts."""

import asyncio
import pickle
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import quote, unquote, urlsplit

import pytest

from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultPayloadStrategy
from plato.clients.transport import InboundTransfer, TransportLimits, receive_s3_payload
from plato.config import Config
from plato.utils.s3 import S3


@pytest.fixture
def local_object_store():
    """A local HTTP contract for the actual S3 writer and bounded reader."""
    objects = {}
    requests = []
    state = {"slow": False}
    stop_trickle = threading.Event()
    get_finished = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def _key(self):
            key = unquote(urlsplit(self.path).path).lstrip("/")
            requests.append((self.command, key))
            return key

        def do_HEAD(self):
            key = self._key()
            self.send_response(200 if key == "bucket" or key in objects else 404)
            self.end_headers()

        def do_PUT(self):
            key = self._key()
            objects[key] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            key = self._key()
            body = objects.get(key)
            try:
                self.send_response(200 if body is not None else 404)
                if not state["slow"]:
                    self.send_header("Content-Length", str(len(body or b"")))
                self.end_headers()
                if body is not None:
                    if state.get("trickle_interval"):
                        for byte in body:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            if stop_trickle.wait(state["trickle_interval"]):
                                break
                    elif state["slow"]:
                        self.wfile.write(body[:1])
                        self.wfile.flush()
                        time.sleep(0.12)
                        self.wfile.write(body[1:])
                    else:
                        self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when the bounded reader expires a slow response.
            finally:
                get_finished.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"

    def adapter(prefix):
        def presign(ClientMethod, Params, ExpiresIn):
            return f"{endpoint}/{Params['Bucket']}/{quote(Params['Key'], safe='/')}"

        return SimpleNamespace(
            key_prefix=prefix,
            bucket="bucket",
            s3_client=SimpleNamespace(generate_presigned_url=presign),
        )

    try:
        yield SimpleNamespace(
            endpoint=endpoint,
            objects=objects,
            requests=requests,
            state=state,
            adapter=adapter,
            stop_trickle=stop_trickle,
            get_finished=get_finished,
        )
    finally:
        stop_trickle.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize("prefix", ["", "prefix", "nested/prefix", "prefix/"])
@pytest.mark.parametrize("minimal_adapter", [False, True])
@pytest.mark.parametrize("kind", ["valid", "oversize", "deadline"])
def test_actual_s3_writer_to_bounded_reader(
    temp_config, local_object_store, prefix, minimal_adapter, kind
):
    """Use real serialization, HTTP upload, presigning and runtime ingress."""
    storage = S3(
        endpoint=local_object_store.endpoint,
        access_key="local-test-key",
        secret_key="local-test-secret",
        bucket="s3://bucket/" + prefix,
    )
    payload = {"w": [1.0, 2.0, 3.0]}
    logical_key = "client_payload_7_ABC123"
    storage.send_to_s3(logical_key, payload)
    canonical = prefix.strip("/")
    object_key = f"{canonical}/{logical_key}" if canonical else logical_key
    physical_key = f"bucket/{object_key}"
    assert set(local_object_store.objects) == {physical_key}
    assert local_object_store.requests == [
        ("HEAD", "bucket"),
        ("HEAD", physical_key),
        ("PUT", physical_key),
    ]
    raw = local_object_store.objects[physical_key]
    if minimal_adapter:
        # Keep the actual botocore presigner, exposing only the supported contract.
        storage = SimpleNamespace(
            key_prefix=prefix, bucket=storage.bucket, s3_client=storage.s3_client
        )
    limits = TransportLimits(
        max_payload_bytes=len(raw) - 1 if kind == "oversize" else 1024,
        payload_timeout=1.0 if kind == "deadline" else 5.0,
    )
    if kind == "deadline":
        local_object_store.state.update(slow=True, trickle_interval=0.05)

    async def scenario():
        transfer = InboundTransfer(limits, lambda: None)
        started = asyncio.get_running_loop().time()
        try:
            # Expiring this outer watchdog fails the test, even for deadline cases.
            async with asyncio.timeout(5.0):
                if kind == "valid":
                    result = await receive_s3_payload(
                        storage, logical_key, transfer, transfer.reserve
                    )
                    assert result == payload
                    assert transfer.byte_count == len(raw)
                elif kind == "oversize":
                    with pytest.raises(ValueError, match="maximum byte limit"):
                        await receive_s3_payload(
                            storage, logical_key, transfer, transfer.reserve
                        )
                    assert transfer.byte_count == 0  # Reject advertised size first.
                else:
                    with pytest.raises(TimeoutError):
                        await receive_s3_payload(
                            storage, logical_key, transfer, transfer.reserve
                        )
                    elapsed = asyncio.get_running_loop().time() - started
                    assert elapsed < limits.payload_timeout + 1.0
                    assert 1 < transfer.byte_count < len(raw)
                assert transfer.byte_count <= limits.max_payload_bytes
        finally:
            # InboundTransfer's caller owns close; the reader owns its HTTP session.
            transfer.close()
            local_object_store.stop_trickle.set()
        assert transfer.closed and transfer.timer.cancelled()
        assert transfer.chunks == [] and transfer.payload is None
        with pytest.raises(ValueError, match="expired or already completed"):
            transfer.check_active()
        await asyncio.sleep(0)
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario(), debug=True)
    assert local_object_store.get_finished.wait(timeout=2)
    assert local_object_store.requests[-1] == ("GET", physical_key)
    assert len(local_object_store.requests) == 4


@pytest.mark.parametrize("prefix", ["", "prefix", "nested/prefix", "/prefix/"])
@pytest.mark.parametrize("qualified", [False, True])
def test_minimal_s3_adapter_prefixes_exactly_once(
    temp_config, local_object_store, prefix, qualified
):
    logical_key = "client_payload_7_ABC123"
    canonical = prefix.strip("/")
    object_key = f"{canonical}/{logical_key}" if canonical else logical_key
    local_object_store.objects[f"bucket/{object_key}"] = pickle.dumps({"w": 7})

    async def scenario():
        transfer = InboundTransfer(TransportLimits(), lambda: None)
        try:
            result = await receive_s3_payload(
                local_object_store.adapter(prefix),
                object_key if qualified else logical_key,
                transfer,
                transfer.reserve,
            )
            assert result == {"w": 7}
            assert transfer.byte_count == len(pickle.dumps(result))
            assert local_object_store.requests == [("GET", f"bucket/{object_key}")]
        finally:
            transfer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("transport", ["socket", "s3"])
def test_client_buffer_cap_applies_below_payload_cap(
    temp_config, local_object_store, transport
):
    Config().server.max_buffered_bytes = 20
    Config().server.max_payload_bytes = Config().server.max_chunk_bytes = 256
    raw = pickle.dumps({"w": "x" * 60})

    async def scenario():
        context = ClientContext()
        context.client_id = 7
        context.current_round = 1
        context.comm_simulation = False
        context.s3_client = local_object_store.adapter("")
        strategy = DefaultPayloadStrategy()
        strategy.reset_payload(context)
        try:
            with pytest.raises(ValueError, match="limit"):
                if transport == "socket":
                    await strategy.accumulate_chunk(context, raw)
                else:
                    # Use a nonempty prefix here so the original prefix bug does not
                    # mask the independently reproduced client buffer-cap defect.
                    context.s3_client = local_object_store.adapter("prefix")
                    local_object_store.objects["bucket/prefix/server_payload_1_1"] = raw
                    await strategy.finalise_inbound_payload(
                        context, 7, s3_key="server_payload_1_1"
                    )
            assert context.chunks == [] and context.server_payload is None
            assert "inbound_transfer" not in context.state
        finally:
            strategy.teardown(context)

    asyncio.run(scenario())
