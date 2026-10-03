"""Cache values and required-image recovery from the second Phase 2A review."""

import contextlib
import http.server
import io
import threading
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from plato.config import Config
from plato.datasources import base, cinic10, purchase, texas, tiny_imagenet
from tests.datasources.test_review_regressions import raw_vector_data


@pytest.mark.parametrize("module", [purchase, texas])
@pytest.mark.parametrize(
    "invalid",
    [
        "string_features",
        "nan_features",
        "inf_features",
        "overflow_features",
        "complex_features",
        "string_labels",
        "fractional_labels",
        "nan_labels",
        "inf_labels",
        "negative_labels",
        "class_overflow",
        "uint64_labels",
        "complex_labels",
    ],
)
def test_bad_cache_values_rebuild_from_raw_without_poisoning_future_startup(
    temp_config, tmp_path, monkeypatch, module, invalid
):
    name = "purchase" if module is purchase else "texas"
    raw_vector_data(tmp_path, name)
    Config.params["data_path"] = str(tmp_path)
    cache = tmp_path / (name + "_numpy.npz")
    features = np.array([[0.1, 0.2], [0.3, 0.4]])
    labels = np.array([0, 1])
    if invalid == "string_features":
        features = np.full((2, 2), "not-numeric")
    elif invalid == "complex_features":
        features = np.full((2, 2), 1 + 2j)
    elif invalid.endswith("features"):
        value = {
            "nan_features": np.nan,
            "inf_features": np.inf,
            "overflow_features": np.finfo(np.float64).max,
        }[invalid]
        features[0, 0] = value
    else:
        labels = {
            "string_labels": np.array(["0", "1"]),
            "fractional_labels": np.array([0.7, 1.7]),
            "nan_labels": np.array([np.nan, 1]),
            "inf_labels": np.array([np.inf, 1]),
            "negative_labels": np.array([-1, 1]),
            "class_overflow": np.array([100, 1]),
            "uint64_labels": np.array([2**64 - 1, 1], dtype=np.uint64),
            "complex_labels": np.array([1 + 2j, 1]),
        }[invalid]
    np.savez(cache, X=features, Y=labels)
    monkeypatch.setattr(
        module.request,
        "urlretrieve",
        lambda *a, **kw: pytest.fail("Complete raw data must recover offline"),
    )
    for _ in range(2):
        source = module.DataSource()
        assert source.num_train_examples() == 2
        assert source.trainset.targets.tolist() == [1, 0]
        torch.testing.assert_close(
            source.trainset.data, torch.tensor([[0.3, 0.4], [0.1, 0.2]])
        )
        with np.load(cache, allow_pickle=False) as rebuilt:
            np.testing.assert_allclose(rebuilt["X"], [[0.1, 0.2], [0.3, 0.4]])
            np.testing.assert_array_equal(rebuilt["Y"], [0, 1])


@pytest.mark.parametrize("module", [purchase, texas])
@pytest.mark.parametrize("numeric", ["native", "big_endian", "bool", "boundary"])
def test_compatible_numeric_cache_only_keeps_values_and_original_file(
    temp_config, tmp_path, monkeypatch, module, numeric
):
    Config.params["data_path"] = str(tmp_path)
    name = "purchase" if module is purchase else "texas"
    cache = tmp_path / (name + "_numpy.npz")
    features = np.array([[0.1, 0.2], [0.3, 0.4]])
    labels = np.array([0.0, 99.0])
    expected = torch.tensor([[0.3, 0.4], [0.1, 0.2]])
    expected_labels = [99, 0]
    if numeric == "big_endian":
        features = features.astype(">f8")
        labels = labels.astype(">i8")
    elif numeric == "bool":
        features = np.array([[False, True], [True, False]])
        labels = np.array([False, True])
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        expected_labels = [1, 0]
    elif numeric == "boundary":
        maximum = np.finfo(np.float32).max
        features = np.array([[maximum, -maximum], [0, 1]], dtype=np.float64)
        labels = np.array([0, 99], dtype=np.uint64)
        expected = torch.tensor([[0, 1], [maximum, -maximum]])
    np.savez(cache, X=features, Y=labels)
    original = cache.read_bytes()
    monkeypatch.setattr(
        module.request,
        "urlretrieve",
        lambda *a, **kw: pytest.fail("Valid cache-only data must work offline"),
    )
    source = module.DataSource()
    assert source.trainset.targets.tolist() == expected_labels
    torch.testing.assert_close(source.trainset.data, expected)
    assert torch.isfinite(source.trainset.data).all()
    assert cache.read_bytes() == original


@contextlib.contextmanager
def counted_archive_server(payload):
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/fixture.zip", hits
    finally:
        server.shutdown()
        worker.join(3)
        server.server_close()
        assert not worker.is_alive()


@pytest.mark.parametrize(
    "module,layout,default_url",
    [
        (cinic10, "prepared", False),
        (tiny_imagenet, "native", False),
        (tiny_imagenet, "native", True),
        (tiny_imagenet, "prepared", False),
    ],
)
@pytest.mark.parametrize("damage", ["evaluation_image", "training_image"])
def test_downloaded_dataset_recovers_missing_sample_with_marker_still_present(
    temp_config, tmp_path, monkeypatch, module, layout, default_url, damage
):
    Config.params["data_path"] = str(tmp_path)
    prefix = "tiny-imagenet-200/" if layout == "native" else ""
    train_file = prefix + "train/n001/images/a.png"
    test_file = prefix + "val/images/z.png" if layout == "native" else "test/n001/z.png"
    png, archive = io.BytesIO(), io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(png, format="PNG")
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(train_file, png.getvalue())
        zipped.writestr(test_file, png.getvalue())
        if layout == "native":
            zipped.writestr(prefix + "val/val_annotations.txt", "z.png\tn001\n")
    actual_get = base.requests.get
    with counted_archive_server(archive.getvalue()) as (url, hits):
        if default_url:

            def redirected_get(requested_url, **kwargs):
                assert requested_url == (
                    "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
                )
                return actual_get(url, **kwargs)

            monkeypatch.setattr(base.requests, "get", redirected_get)
        else:
            Config.data.download_url = url
        initial = module.DataSource()
        assert initial.testset[0][1] == 0
        assert hits == ["/fixture.zip"]
        markers = list(tmp_path.glob("*.complete"))
        assert len(markers) == 1
        missing = tmp_path / (test_file if damage == "evaluation_image" else train_file)
        missing.unlink()
        assert missing.parent.is_dir() and markers[0].is_file()
        recovered = module.DataSource()
        assert missing.is_file()
        assert hits == ["/fixture.zip", "/fixture.zip"]
        assert recovered.trainset[0][1] == recovered.testset[0][1] == 0
        assert torch.isfinite(recovered.testset[0][0]).all()
        assert markers[0].is_file()
    monkeypatch.setattr(
        base.requests,
        "get",
        lambda *a, **kw: pytest.fail("Complete roots must stay offline"),
    )
    offline = module.DataSource()
    assert offline.testset[0][1] == offline.trainset[0][1] == 0
