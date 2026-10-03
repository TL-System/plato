"""Failed and concurrent default Tiny recovery with actual HTTP and flock."""

import contextlib
import http.server
import io
import sys
import threading
import zipfile

import pytest
import torch
from PIL import Image

from plato.config import Config
from plato.datasources import base, tiny_imagenet
from tests.datasources.test_local_datasets import image_file

DEFAULT_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
MISSING_IMAGE = "tiny-imagenet-200/val/images/second.PNG"


def native_members():
    png = io.BytesIO()
    Image.new("RGB", (19, 13), (40, 90, 160)).save(png, format="PNG")
    return {
        "tiny-imagenet-200/train/n001/images/a.PNG": png.getvalue(),
        "tiny-imagenet-200/train/n002/images/b.PNG": png.getvalue(),
        "tiny-imagenet-200/val/images/first.PNG": png.getvalue(),
        MISSING_IMAGE: png.getvalue(),
        "tiny-imagenet-200/val/val_annotations.txt": (
            b"first.PNG\tn002\nsecond.PNG\tn002\n"
        ),
    }


def zip_bytes(members):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for filename, contents in members.items():
            archive.writestr(filename, contents)
    return output.getvalue()


@contextlib.contextmanager
def recovery_server(payload):
    state = {
        "payload": payload,
        "hits": [],
        "pause": False,
        "entered": threading.Event(),
        "release": threading.Event(),
    }

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = state["payload"]
            state["hits"].append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state["pause"]:
                state["entered"].set()
                if not state["release"].wait(10):
                    return
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/fixture.zip", state
    finally:
        state["release"].set()
        server.shutdown()
        worker.join(3)
        server.server_close()
        assert not worker.is_alive()


def configure_source(tmp_path, monkeypatch, url, default):
    Config.params["data_path"] = str(tmp_path)
    if default:
        actual_get = base.requests.get

        def redirected_get(requested_url, **kwargs):
            assert requested_url == DEFAULT_URL
            return actual_get(url, **kwargs)

        monkeypatch.setattr(base.requests, "get", redirected_get)
    else:
        Config.data.download_url = url


def check_samples(source):
    assert source.classes() == ["n001", "n002"]
    assert (
        source.trainset.class_to_idx
        == source.testset.class_to_idx
        == {
            "n001": 0,
            "n002": 1,
        }
    )
    assert source.trainset.targets == [0, 1]
    assert source.testset.targets == [1, 1]
    for dataset in (source.trainset, source.testset):
        for index in range(len(dataset)):
            tensor, label = dataset[index]
            assert label == dataset.targets[index]
            assert tensor.shape == (3, 299, 299) and torch.isfinite(tensor).all()
    assert torch.equal(source.testset[0][0], source.testset[0][0])


@pytest.mark.parametrize("default", [True, False])
def test_tiny_failed_recovery_retries_restored_source_and_stays_offline(
    temp_config, tmp_path, monkeypatch, default
):
    members = native_members()
    missing = tmp_path / MISSING_IMAGE
    with recovery_server(zip_bytes(members)) as (url, state):
        configure_source(tmp_path, monkeypatch, url, default)
        check_samples(tiny_imagenet.DataSource())
        missing.unlink()
        state["payload"] = zip_bytes(
            {name: value for name, value in members.items() if name != MISSING_IMAGE}
        )
        with pytest.raises(RuntimeError, match="required extracted dataset artifacts"):
            tiny_imagenet.DataSource()
        assert state["hits"] == ["/fixture.zip"] * 2
        assert not missing.exists() and not list(tmp_path.glob("*.complete"))
        if default:
            assert (tmp_path / "tiny-imagenet-200.zip").is_file()
        state["payload"] = zip_bytes(members)
        check_samples(tiny_imagenet.DataSource())
        assert missing.read_bytes() == members[MISSING_IMAGE]
        assert state["hits"] == ["/fixture.zip"] * 3
        assert len(list(tmp_path.glob("*.complete"))) == 1
    monkeypatch.setattr(
        base.requests,
        "get",
        lambda *a, **kw: pytest.fail("Complete roots stay offline"),
    )
    check_samples(tiny_imagenet.DataSource())


@pytest.mark.parametrize("default", [True, False])
def test_tiny_contender_waits_for_paused_recovery_and_reuses_completed_root(
    temp_config, tmp_path, monkeypatch, default
):
    members = native_members()
    values, errors = [], []
    contender_entered, contender_finished = threading.Event(), threading.Event()

    def load(contender=False):
        def observe_call(frame, event, function):
            if event == "c_call" and function is base.fcntl.flock:
                contender_entered.set()

        if contender:
            # Observe the unmodified C call, then let the real flock block.
            # A rejected constructor signals from finally instead. No sleeps
            # or patched lock/download/readiness functions gate the outcome.
            sys.setprofile(observe_call)
        try:
            values.append(tiny_imagenet.DataSource())
        except Exception as error:
            errors.append(error)
        finally:
            if contender:
                sys.setprofile(None)
                contender_finished.set()
                contender_entered.set()

    with recovery_server(zip_bytes(members)) as (url, state):
        configure_source(tmp_path, monkeypatch, url, default)
        check_samples(tiny_imagenet.DataSource())
        (tmp_path / MISSING_IMAGE).unlink()
        state["pause"] = True
        workers = [
            threading.Thread(target=load),
            threading.Thread(target=load, kwargs={"contender": True}),
        ]
        try:
            workers[0].start()
            assert state["entered"].wait(5), "Owner did not reach paused HTTP body"
            assert not list(tmp_path.glob("*.complete"))
            workers[1].start()
            assert contender_entered.wait(5), "Contender never reached flock or failed"
            assert not contender_finished.is_set(), (
                f"Contender rejected live recovery before waiting: {errors!r}"
            )
            assert state["hits"] == ["/fixture.zip"] * 2
        finally:
            state["release"].set()
            for worker in workers:
                if worker.ident is not None:
                    worker.join(10)
        assert not any(worker.is_alive() for worker in workers)
        assert not errors, errors
        assert len(values) == 2
        assert state["hits"] == ["/fixture.zip"] * 2
        assert len(list(tmp_path.glob("*.complete"))) == 1
        for source in values:
            check_samples(source)
    monkeypatch.setattr(
        base.requests,
        "get",
        lambda *a, **kw: pytest.fail("Complete roots stay offline"),
    )
    check_samples(tiny_imagenet.DataSource())


@pytest.mark.parametrize("artifact", [None, ".download.lock", "unrelated.zip"])
def test_incomplete_manual_native_root_cannot_use_unrelated_recovery_artifact(
    temp_config, tmp_path, monkeypatch, artifact
):
    Config.params["data_path"] = str(tmp_path)
    image_file(tmp_path / "tiny-imagenet-200/train/n001/images/a.png")
    image_file(tmp_path / "tiny-imagenet-200/test/images/unlabeled.png")
    if artifact is not None:
        (tmp_path / artifact).write_bytes(b"unrelated")
    monkeypatch.setattr(
        base.requests, "get", lambda *a, **kw: pytest.fail("Manual incomplete root")
    )
    with pytest.raises(ValueError, match="Incomplete native.*provide a download_url"):
        tiny_imagenet.DataSource()
