"""Default recovery eligibility after real HTTP and archive download errors."""

import contextlib
import http.server
import threading

import pytest

from plato.datasources import base, tiny_imagenet
from tests.datasources.test_review3_regressions import (
    MISSING_IMAGE,
    check_samples,
    configure_source,
    native_members,
    zip_bytes,
)


@contextlib.contextmanager
def failing_source(payload):
    state = {"payload": payload, "status": 200, "truncated": False, "hits": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body, status = state["payload"], state["status"]
            state["hits"].append((self.path, status))
            self.send_response(status)
            self.send_header(
                "Content-Length", str(len(body) + (7 if state["truncated"] else 0))
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/fixture.zip", state
    finally:
        server.shutdown()
        worker.join(3)
        server.server_close()
        assert not worker.is_alive()


def fail_response(state, failure):
    if failure == "http_503":
        state["status"] = 503
        return base.requests.HTTPError
    if failure == "bad_zip":
        state["payload"] = b"not a ZIP archive"
        return base.tarfile.ReadError
    state["truncated"] = True
    return base.requests.exceptions.ChunkedEncodingError


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("failure", ["http_503", "bad_zip", "truncated_body"])
def test_tiny_download_error_keeps_default_recovery_eligible_with_restored_source(
    temp_config, tmp_path, monkeypatch, default, failure
):
    members = native_members()
    payload = zip_bytes(members)
    with failing_source(payload) as (url, state):
        configure_source(tmp_path, monkeypatch, url, default)
        check_samples(tiny_imagenet.DataSource())
        canonical = tmp_path / ("tiny-imagenet-200.zip" if default else "fixture.zip")
        assert canonical.read_bytes() == payload
        (tmp_path / MISSING_IMAGE).unlink()
        expected_error = fail_response(state, failure)
        with pytest.raises(expected_error):
            tiny_imagenet.DataSource()
        assert len(state["hits"]) == 4  # Initial success plus three normal retries.
        assert not list(tmp_path.glob("*.complete"))
        assert not (tmp_path / MISSING_IMAGE).exists()
        retained = canonical.read_bytes() if canonical.is_file() else None
        assert not list(tmp_path.glob(".download-*"))
        state.update(payload=payload, status=200, truncated=False)
        # Actual constructor is the negative path: old default code refuses
        # before requesting this restored source. Configured controls recover.
        check_samples(tiny_imagenet.DataSource())
        assert len(state["hits"]) == 5 and state["hits"][-1] == ("/fixture.zip", 200)
        assert (tmp_path / MISSING_IMAGE).read_bytes() == members[MISSING_IMAGE]
        assert canonical.read_bytes() == payload
        if default:
            assert retained == payload
        assert len(list(tmp_path.glob("*.complete"))) == 1
        assert not list(tmp_path.glob(".download-*"))
    monkeypatch.setattr(
        base.requests,
        "get",
        lambda *a, **kw: pytest.fail("Complete roots stay offline"),
    )
    check_samples(tiny_imagenet.DataSource())


@pytest.mark.parametrize("failure", ["http_503", "bad_zip", "truncated_body"])
def test_failed_initial_default_download_does_not_publish_an_archive_or_completion(
    temp_config, tmp_path, monkeypatch, failure
):
    payload = zip_bytes(native_members())
    with failing_source(payload) as (url, state):
        configure_source(tmp_path, monkeypatch, url, True)
        expected_error = fail_response(state, failure)
        with pytest.raises(expected_error):
            tiny_imagenet.DataSource()
        assert len(state["hits"]) == 3
        assert not (tmp_path / "tiny-imagenet-200.zip").exists()
        assert not list(tmp_path.glob("*.complete"))
        assert not list(tmp_path.glob(".download-*"))
        state.update(payload=payload, status=200, truncated=False)
        check_samples(tiny_imagenet.DataSource())
        assert len(state["hits"]) == 4
        assert (tmp_path / "tiny-imagenet-200.zip").read_bytes() == payload
        assert len(list(tmp_path.glob("*.complete"))) == 1
        assert not list(tmp_path.glob(".download-*"))
