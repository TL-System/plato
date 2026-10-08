"""Public S3 methods against an isolated, credential-free HTTP service."""

import pickle
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse
from xml.sax.saxutils import escape

import pytest
from botocore.exceptions import ClientError

from plato.utils import s3


@pytest.fixture
def service(monkeypatch):
    state = SimpleNamespace(
        objects={},
        calls=Counter(),
        fault=None,
        release=threading.Event(),
        prefixes=[],
        transient_failures=0,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *_args):
            pass

        def respond(self, status, body=b""):
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def handle_request(self):
            parsed = urlparse(self.path)
            key = unquote(parsed.path).removeprefix("/bucket/")
            state.calls[(self.command, key)] += 1
            if state.transient_failures:
                state.transient_failures -= 1
                return self.respond(503)
            if state.fault == "stall":
                state.release.wait(4)
            if state.fault == "error":
                return self.respond(503)
            if state.fault == "forbidden":
                return self.respond(403)
            if parsed.path == "/bucket" and self.command == "HEAD":
                return self.respond(200)
            if self.command == "PUT":
                state.objects[key] = self.rfile.read(
                    int(self.headers["Content-Length"])
                )
                return self.respond(200)
            if self.command == "DELETE":
                state.objects.pop(key, None)
                return self.respond(204)
            query = parse_qs(parsed.query)
            if query.get("list-type") == ["2"]:
                prefix = query.get("prefix", [""])[0]
                state.prefixes.append(prefix)
                keys = sorted(key for key in state.objects if key.startswith(prefix))
                offset = int(query.get("continuation-token", ["0"])[0])
                page = keys[offset : offset + 2]
                more = offset + 2 < len(keys)
                contents = "".join(
                    f"<Contents><Key>{escape(key)}</Key></Contents>" for key in page
                )
                token = (
                    f"<NextContinuationToken>{offset + 2}</NextContinuationToken>"
                    if more
                    else ""
                )
                xml = (
                    '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                    f"<IsTruncated>{str(more).lower()}</IsTruncated>{contents}{token}"
                    "</ListBucketResult>"
                )
                return self.respond(200, xml.encode())
            if key not in state.objects:
                return self.respond(404)
            return self.respond(200, state.objects[key])

        do_HEAD = handle_request
        do_PUT = handle_request
        do_GET = handle_request
        do_DELETE = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(s3, "Config", lambda: SimpleNamespace(server=SimpleNamespace()))
    monkeypatch.setattr(s3.S3, "NETWORK_TIMEOUT", (0.1, 0.1), raising=False)
    monkeypatch.setattr(s3.S3, "MAX_ATTEMPTS", 2, raising=False)
    state.endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


def storage(service, prefix=""):
    return s3.S3(
        endpoint=service.endpoint,
        access_key="local-test-key",
        secret_key="local-test-secret",
        bucket="s3://bucket/" + prefix,
    )


@pytest.mark.parametrize("prefix", ["", "namespace/rounds", "namespace/rounds/"])
def test_put_send_receive_list_delete_use_one_namespace(service, prefix):
    client = storage(service, prefix)
    expected = (prefix.strip("/") + "/" if prefix.strip("/") else "") + "model"
    client.send_to_s3("model", {"weight": [1, 2]})
    assert set(service.objects) == {expected}
    assert client.receive_from_s3("model") == {"weight": [1, 2]}
    client.send_to_s3("model", {"weight": [3]})
    assert client.receive_from_s3("model") == {"weight": [1, 2]}
    client.put_to_s3("model", {"weight": [4]})
    assert client.receive_from_s3("model") == {"weight": [4]}
    assert client.lists() == [expected]
    client.delete_from_s3(client.lists()[0])
    assert service.objects == {}
    client.put_to_s3("model", 5)
    client.delete_from_s3("model")
    assert client.lists() == []


def test_empty_listing_succeeds_without_contents(service):
    client = storage(service, "namespace")
    assert client.lists() == []


def test_all_pages_stay_in_namespace(service):
    client = storage(service, "namespace")
    service.objects = {
        **{f"namespace/{key}": pickle.dumps(key) for key in range(5)},
        "namespace-other/unrelated": b"preserve",
        "elsewhere": b"preserve",
    }
    assert client.lists() == [f"namespace/{key}" for key in range(5)]
    assert service.prefixes == ["namespace/"] * 3


def test_service_errors_do_not_create_or_overwrite_objects(service):
    client = storage(service)
    service.fault = "forbidden"
    with pytest.raises(ClientError):
        client.send_to_s3("model", 1)
    assert not any(method == "PUT" for method, _key in service.calls)
    with pytest.raises(ClientError):
        client.lists()
    with pytest.raises(ClientError):
        storage(service)


@pytest.mark.parametrize("fault", ["stall", "error"])
@pytest.mark.parametrize("operation", ["put", "receive", "list", "delete", "init"])
def test_network_paths_fail_within_deadline(service, fault, operation):
    client = storage(service)
    service.fault = fault
    errors = []

    def invoke():
        try:
            if operation == "init":
                storage(service)
            elif operation == "list":
                client.lists()
            else:
                method = getattr(client, operation + "_from_s3", None)
                if operation == "put":
                    client.put_to_s3("model", 1)
                else:
                    assert callable(method)
                    method("model")
        except Exception as exc:
            errors.append(exc)

    started = time.monotonic()
    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    worker.join(2.5)
    try:
        assert not worker.is_alive(), f"{operation} exceeded the 2.5-second deadline"
        assert errors, f"{operation} silently succeeded on {fault}"
        assert time.monotonic() - started < 2.5
        # Finite attempts on actual HTTP paths, including HEAD and DELETE.
        assert max(service.calls.values()) <= 3  # initial successful HEAD + 2 retries
    finally:
        service.release.set()
        worker.join(5)


def test_truncated_pickle_failure_is_visible(service):
    client = storage(service)
    service.objects["model"] = pickle.dumps({"w": 1})[:-1]
    with pytest.raises((pickle.UnpicklingError, EOFError)):
        client.receive_from_s3("model")


def test_successful_transfer_recovers_after_transient_service_error(service):
    client = storage(service)
    service.transient_failures = 1
    client.put_to_s3("model", {"w": [2, 4]})
    assert service.calls[("PUT", "model")] == 2
    service.transient_failures = 1
    assert client.receive_from_s3("model") == {"w": [2, 4]}
    assert service.calls[("GET", "model")] == 2
    service.transient_failures = 1
    assert client.lists() == ["model"]


def test_missing_object_is_distinct_from_service_failure(service):
    client = storage(service)
    with pytest.raises(FileNotFoundError):
        client.receive_from_s3("missing")
