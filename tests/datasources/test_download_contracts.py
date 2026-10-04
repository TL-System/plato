"""Inert archive containment and download lifecycle tests on active entrypoints."""

import contextlib
import gzip
import http.server
import io
import selectors
import shutil
import subprocess
import sys
import tarfile
import threading
import zipfile

import pytest

from plato.datasources import base, purchase, texas


@contextlib.contextmanager
def serve(payload, filename):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/{filename}"
    finally:
        server.shutdown()
        worker.join(2)
        server.server_close()
        assert not worker.is_alive()


def archive_bytes(kind, members):
    stream = io.BytesIO()
    if kind == "zip":
        with zipfile.ZipFile(stream, "w") as archive:
            for name, value in members:
                archive.writestr(name, value)
    else:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name, value in members:
                info = tarfile.TarInfo(name)
                if isinstance(value, tuple):
                    info.type, info.linkname = value
                    archive.addfile(info)
                else:
                    info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
    return stream.getvalue()


def invoke_download(entrypoint, payload, destination, tmp_path, monkeypatch):
    if entrypoint.startswith("base"):
        suffix = "zip" if entrypoint.endswith("zip") else "tar.gz"
        with serve(payload, "fixture." + suffix) as url:
            base.DataSource.download(url, str(destination))
    else:
        archive = tmp_path / "fixture.tgz"
        archive.write_bytes(payload)
        module = purchase if entrypoint == "purchase" else texas
        monkeypatch.setattr(
            module.request,
            "urlretrieve",
            lambda url, target: shutil.copyfile(archive, target),
        )
        if entrypoint == "purchase":
            source = purchase.DataSource.__new__(purchase.DataSource)
            source.download_dataset(str(destination), destination / "dataset_purchase")
        else:
            source = texas.DataSource.__new__(texas.DataSource)
            source.download_dataset(
                str(destination),
                destination / "texas/100/feats",
                destination / "texas/100/labels",
            )


@pytest.mark.parametrize("entrypoint", ["base_tar", "base_zip", "purchase", "texas"])
@pytest.mark.parametrize(
    "attack", ["traversal", "absolute", "existing_directory_link", "existing_file_link"]
)
def test_active_extractors_reject_unsafe_paths_before_writes(
    tmp_path, monkeypatch, entrypoint, attack
):
    destination = tmp_path / "download"
    outside = tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    escaped = outside / "escaped.txt"
    escaped.write_bytes(b"original")
    if attack == "traversal":
        name = "../outside/escaped.txt"
    elif attack == "absolute":
        name = str(escaped)
    elif attack == "existing_directory_link":
        (destination / "link").symlink_to(outside, target_is_directory=True)
        name = "link/escaped.txt"
    else:
        (destination / "link").symlink_to(escaped)
        name = "link"
    kind = "zip" if entrypoint == "base_zip" else "tar"
    payload = archive_bytes(kind, [("safe.txt", b"safe"), (name, b"changed")])
    monkeypatch.setattr(base.time, "sleep", lambda seconds: None)
    with pytest.raises(
        (ValueError, tarfile.FilterError),
        match="(?i)unsafe|outside|link|absolute|traversal",
    ):
        invoke_download(entrypoint, payload, destination, tmp_path, monkeypatch)
    assert escaped.read_bytes() == b"original"
    assert not (destination / "safe.txt").exists()
    assert not list(destination.glob("*.complete"))
    assert not list(destination.glob("*_numpy.npz"))


@pytest.mark.parametrize("entrypoint", ["base_tar", "purchase", "texas"])
@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_active_tar_extractors_reject_link_escapes(
    tmp_path, monkeypatch, entrypoint, link_type
):
    destination = tmp_path / "download"
    outside = tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    escaped = outside / "escaped.txt"
    escaped.write_bytes(b"original")
    members = [("safe.txt", b"safe"), ("link", (link_type, "../outside/escaped.txt"))]
    if link_type == tarfile.SYMTYPE:
        # A subsequent file would follow an escaped directory symlink.
        members[1] = ("link", (link_type, "../outside"))
        members.append(("link/escaped.txt", b"changed"))
    monkeypatch.setattr(base.time, "sleep", lambda seconds: None)
    with pytest.raises(
        (ValueError, tarfile.FilterError), match="(?i)unsafe|outside|link"
    ):
        invoke_download(
            entrypoint,
            archive_bytes("tar", members),
            destination,
            tmp_path,
            monkeypatch,
        )
    assert escaped.read_bytes() == b"original"
    assert not (destination / "safe.txt").exists()
    assert not list(destination.glob("*.complete"))
    assert not list(destination.glob("*_numpy.npz"))


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_base_download_safe_archive_and_completion_cache(tmp_path, kind):
    filename = "fixture.tar.gz" if kind == "tar" else "fixture.zip"
    with serve(
        archive_bytes(kind, [("nested/data.txt", b"local dataset")]), filename
    ) as url:
        base.DataSource.download(url, str(tmp_path))
    assert (tmp_path / "nested/data.txt").read_bytes() == b"local dataset"
    assert (tmp_path / (filename + ".complete")).is_file()
    # The HTTP server has stopped: completion must skip the network entirely.
    base.DataSource.download(url, str(tmp_path))


def test_download_guard_recovers_after_owner_process_exits(tmp_path):
    script = """
import os, sys
from plato.datasources.base import DataSource
with DataSource._download_guard(sys.argv[1]):
    print('acquired', flush=True)
    if sys.argv[2] == 'die': os._exit(0)
"""
    first = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), "die"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert first.returncode == 0 and "acquired" in first.stdout
    try:
        second = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), "exit"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("A dead download owner left a permanent lock stall")
    assert second.returncode == 0, second.stderr
    assert "acquired" in second.stdout


def test_failed_download_releases_lock_and_never_marks_complete(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise base.requests.ConnectionError("local connection refused")

    monkeypatch.setattr(base.requests, "get", fail)
    monkeypatch.setattr(base.time, "sleep", lambda seconds: None)
    with pytest.raises(base.requests.ConnectionError):
        base.DataSource.download("http://127.0.0.1/fixture.zip", str(tmp_path))
    assert not list(tmp_path.glob("*.complete"))
    with base.DataSource._download_guard(str(tmp_path)):
        pass


def test_unsupported_download_format_raises_useful_error(tmp_path):
    with serve(b"dataset", "fixture.unsupported") as url:
        with pytest.raises(ValueError, match="Unsupported.*format"):
            base.DataSource.download(url, str(tmp_path))
    assert not list(tmp_path.glob("*.complete"))


@pytest.mark.parametrize("artifact", ["archive", "completion", "gzip_output"])
def test_download_artifact_symlinks_cannot_write_outside(tmp_path, artifact):
    destination = tmp_path / "download"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"original")
    filename = "fixture.gz" if artifact == "gzip_output" else "fixture.zip"
    payload = (
        gzip.compress(b"data")
        if artifact == "gzip_output"
        else archive_bytes("zip", [("safe.txt", b"safe")])
    )
    symlink = {
        "archive": filename,
        "completion": filename + ".complete",
        "gzip_output": "fixture",
    }[artifact]
    target = outside if artifact != "completion" else tmp_path / "new_outside"
    (destination / symlink).symlink_to(target)
    with serve(payload, filename) as url:
        with pytest.raises(ValueError, match="symlink"):
            base.DataSource.download(url, str(destination))
    assert outside.read_bytes() == b"original"
    assert not (tmp_path / "new_outside").exists()


def test_http_error_closes_each_response_before_retry(tmp_path, monkeypatch):
    streams = []

    def failed_response(*args, **kwargs):
        response = base.requests.Response()
        response.status_code = 500
        response.url = "https://fixture.invalid/data.zip"
        response.raw = io.BytesIO(b"server error")
        streams.append(response.raw)
        return response

    monkeypatch.setattr(base.requests, "get", failed_response)
    monkeypatch.setattr(base.time, "sleep", lambda seconds: None)
    with pytest.raises(base.requests.HTTPError, match="500"):
        base.DataSource.download("https://fixture.invalid/data.zip", str(tmp_path))
    assert len(streams) == 3
    assert all(stream.closed for stream in streams)
    assert not list(tmp_path.glob("*.complete"))


def test_interrupted_download_retries_closes_and_removes_partial_file(
    tmp_path, monkeypatch
):
    class BrokenBody(io.BytesIO):
        def read(self, *args):
            if self.tell():
                raise base.requests.exceptions.ChunkedEncodingError(
                    "stream interrupted"
                )
            return super().read(*args)

    bodies = []

    def interrupted_response(*args, **kwargs):
        response = base.requests.Response()
        response.status_code = 200
        response.headers["Content-Length"] = "20"
        response.raw = BrokenBody(b"partial")
        bodies.append(response.raw)
        return response

    monkeypatch.setattr(base.requests, "get", interrupted_response)
    monkeypatch.setattr(base.time, "sleep", lambda seconds: None)
    with pytest.raises(
        base.requests.exceptions.ChunkedEncodingError, match="interrupted"
    ):
        base.DataSource.download("https://fixture.invalid/data.zip", str(tmp_path))
    assert len(bodies) == 3
    assert all(body.closed for body in bodies)
    assert not (tmp_path / "data.zip").exists()
    assert not list(tmp_path.glob("*.complete"))


def test_download_guard_rejects_dangling_lock_symlink(tmp_path):
    outside = tmp_path / "outside"
    destination = tmp_path / "download"
    destination.mkdir()
    (destination / ".download.lock").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        with base.DataSource._download_guard(str(destination)):
            pass
    assert not outside.exists()


def test_download_guard_preserves_live_owner_exclusion(tmp_path):
    script = """
import sys
from plato.datasources.base import DataSource
print('ready', flush=True)
with DataSource._download_guard(sys.argv[1]):
    print('acquired', flush=True)
"""
    child = None
    try:
        with base.DataSource._download_guard(str(tmp_path)):
            child = subprocess.Popen(
                [sys.executable, "-c", script, str(tmp_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with selectors.DefaultSelector() as selector:
                assert child.stdout is not None
                selector.register(child.stdout, selectors.EVENT_READ)
                assert selector.select(timeout=5), "Download contender did not start"
                assert child.stdout.readline() == b"ready\n"
            with pytest.raises(subprocess.TimeoutExpired) as waiting:
                child.communicate(timeout=0.5)
            assert b"acquired" not in (waiting.value.output or b"")
            assert child.poll() is None
        output, errors = child.communicate(timeout=5)
        assert child.returncode == 0, errors.decode()
        assert b"acquired" in output
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


@pytest.mark.parametrize("entrypoint", ["purchase", "texas"])
def test_vector_download_archive_symlink_cannot_write_outside(
    tmp_path, monkeypatch, entrypoint
):
    destination = tmp_path / "download"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"original")
    filename = "tmp_purchase.tgz" if entrypoint == "purchase" else "tmp_texas.tgz"
    (destination / filename).symlink_to(outside)
    payload = archive_bytes("tar", [("safe.txt", b"safe")])
    with pytest.raises(ValueError, match="symlink"):
        invoke_download(entrypoint, payload, destination, tmp_path, monkeypatch)
    assert outside.read_bytes() == b"original"


@pytest.mark.parametrize("entrypoint", ["purchase", "texas"])
def test_vector_download_safe_archive_builds_cache(tmp_path, monkeypatch, entrypoint):
    destination = tmp_path / "download"
    destination.mkdir()
    if entrypoint == "purchase":
        members = [("dataset_purchase", b"1,0.1,0.2\n2,0.3,0.4\n")]
        cache = destination / "purchase_numpy.npz"
    else:
        members = [
            ("texas/100/feats", b"0.1,0.2\n0.3,0.4\n"),
            ("texas/100/labels", b"1\n2\n"),
        ]
        cache = destination / "texas_numpy.npz"
    invoke_download(
        entrypoint, archive_bytes("tar", members), destination, tmp_path, monkeypatch
    )
    import numpy as np

    with np.load(cache) as data:
        np.testing.assert_allclose(data["X"], [[0.1, 0.2], [0.3, 0.4]])
        np.testing.assert_array_equal(data["Y"], [0, 1])
