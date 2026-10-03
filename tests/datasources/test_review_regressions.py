"""Actual startup, publication and evaluation regressions from Phase 2A review."""

import io
import selectors
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from plato.config import Config
from plato.datasources import base, cinic10, purchase, texas, tiny_imagenet
from tests.datasources.test_download_contracts import serve
from tests.datasources.test_local_datasets import image_file


def raw_vector_data(root, name):
    if name == "purchase":
        (root / "dataset_purchase").write_text("1,0.1,0.2\n2,0.3,0.4\n")
    else:
        raw = root / "texas/100"
        raw.mkdir(parents=True)
        (raw / "feats").write_text("0.1,0.2\n0.3,0.4\n")
        (raw / "labels").write_text("1\n2\n")


_WRITER = """
import contextlib, sys
import numpy as np
from plato.config import Config
from plato.datasources import purchase, texas
Config._instance=object.__new__(Config)
Config.data=Config.node_from_dict({})
Config.params={'data_path':sys.argv[1]}
original=np.savez
class PausedFile:
    def __init__(self, handle): self.handle=handle; self.first=True
    def __getattr__(self, name): return getattr(self.handle,name)
    def write(self, value):
        written=self.handle.write(value)
        self.handle.flush()
        if self.first:
            self.first=False
            print('npz-header-written',flush=True)
            sys.stdin.readline()
        return written
def save(file, **kwargs):
    context=(contextlib.nullcontext(file) if hasattr(file,'write')
             else open(file,'wb'))
    with context as handle:
        original(PausedFile(handle),**kwargs)
np.savez=save
module=purchase if sys.argv[2]=='purchase' else texas
source=module.DataSource()
assert source.num_train_examples()==2
print('built',flush=True)
"""

_READER = """
import sys
from plato.config import Config
from plato.datasources import purchase, texas
Config._instance=object.__new__(Config)
Config.data=Config.node_from_dict({})
Config.params={'data_path':sys.argv[1]}
module=purchase if sys.argv[2]=='purchase' else texas
print('reader-started',flush=True)
source=module.DataSource()
assert source.num_train_examples()==2
assert sorted(source.targets().tolist())==[0,1]
print('loaded',flush=True)
"""


def await_line(process, expected):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(10), f"No subprocess startup: {expected}"
        assert process.stdout.readline().strip() == expected


@pytest.mark.parametrize("name", ["purchase", "texas"])
@pytest.mark.parametrize("kill_writer", [False, True])
def test_vector_cache_reader_waits_for_real_writer_or_recovers_after_death(
    temp_config, tmp_path, monkeypatch, name, kill_writer
):
    raw_vector_data(tmp_path, name)
    Config.params["data_path"] = str(tmp_path)
    module = purchase if name == "purchase" else texas
    monkeypatch.setattr(
        module.request,
        "urlretrieve",
        lambda *a, **kw: pytest.fail("Complete raw files must work offline"),
    )
    writer = subprocess.Popen(
        [sys.executable, "-c", _WRITER, str(tmp_path), name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    reader = None
    try:
        await_line(writer, "npz-header-written")
        if kill_writer:
            writer.kill()
            writer.communicate(timeout=10)
            reader = subprocess.Popen(
                [sys.executable, "-c", _READER, str(tmp_path), name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, err = reader.communicate(timeout=10)
            assert reader.returncode == 0, err
            assert "loaded" in out
        else:
            reader = subprocess.Popen(
                [sys.executable, "-c", _READER, str(tmp_path), name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            await_line(reader, "reader-started")
            try:
                out, err = reader.communicate(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            else:
                pytest.fail(f"Reader bypassed live writer: {out} {err}")
            assert not (tmp_path / (name + "_numpy.npz")).exists()
            out, err = writer.communicate("\n", timeout=10)
            assert writer.returncode == 0, err
            assert "built" in out
            out, err = reader.communicate(timeout=10)
            assert reader.returncode == 0, err
            assert "loaded" in out
        source = module.DataSource()
        assert source.num_train_examples() == 2
        assert sorted(source.targets().tolist()) == [0, 1]
        with np.load(tmp_path / (name + "_numpy.npz")) as cache:
            np.testing.assert_allclose(cache["X"], [[0.1, 0.2], [0.3, 0.4]])
            np.testing.assert_array_equal(cache["Y"], [0, 1])
        assert not list(tmp_path.glob(".*_numpy.npz.*.tmp"))
    finally:
        for process in (writer, reader):
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate(timeout=10)


@pytest.mark.parametrize("module", [purchase, texas])
@pytest.mark.parametrize("invalid", ["truncated", "missing_labels", "bad_shape"])
def test_invalid_vector_cache_rebuilds_from_complete_raw_data_offline(
    temp_config, tmp_path, monkeypatch, module, invalid
):
    name = "purchase" if module is purchase else "texas"
    raw_vector_data(tmp_path, name)
    Config.params["data_path"] = str(tmp_path)
    cache = tmp_path / (name + "_numpy.npz")
    if invalid == "truncated":
        cache.write_bytes(b"PK\x03\x04\x00\x00")
    elif invalid == "missing_labels":
        np.savez(cache, X=np.ones((2, 2)))
    else:
        np.savez(cache, X=np.ones((2, 2)), Y=np.array([0]))
    monkeypatch.setattr(
        module.request,
        "urlretrieve",
        lambda *a, **kw: pytest.fail("Cache recovery must reuse complete raw data"),
    )
    source = module.DataSource()
    assert source.num_train_examples() == 2
    assert sorted(source.targets().tolist()) == [0, 1]


@pytest.mark.parametrize("module", [cinic10, tiny_imagenet])
def test_prepared_evaluation_subset_uses_training_labels_and_metadata(
    temp_config, tmp_path, module
):
    Config.params["data_path"] = str(tmp_path)
    image_file(tmp_path / "train/n001/one.png")
    image_file(tmp_path / "train/n002/two.png")
    image_file(tmp_path / "test/n002/evaluate.png")
    source = module.DataSource()
    train, test = source.trainset, source.testset
    assert train.class_to_idx == test.class_to_idx == {"n001": 0, "n002": 1}
    assert train.classes == test.classes == ["n001", "n002"]
    assert test.targets == [1]
    assert test.samples == test.imgs == [(str(tmp_path / "test/n002/evaluate.png"), 1)]
    assert test[0][1] == 1
    assert source.num_train_examples() == 2
    assert source.num_test_examples() == 1


@pytest.mark.parametrize("module", [cinic10, tiny_imagenet])
def test_prepared_evaluation_rejects_unknown_classes(temp_config, tmp_path, module):
    Config.params["data_path"] = str(tmp_path)
    image_file(tmp_path / "train/n001/one.png")
    image_file(tmp_path / "test/n999/unknown.png")
    with pytest.raises(ValueError, match="(?i)unknown.*class|class.*unknown"):
        module.DataSource()


@pytest.mark.parametrize("nested", [False, True])
def test_incomplete_native_tiny_never_fabricates_test_images_class(
    temp_config, tmp_path, monkeypatch, nested
):
    Config.params["data_path"] = str(tmp_path)
    root = tmp_path / "tiny-imagenet-200" if nested else tmp_path
    image_file(root / "train/n001/images/one.png")
    image_file(root / "train/n002/images/two.png")
    image_file(root / "test/images/unlabeled.png")
    monkeypatch.setattr(
        tiny_imagenet.DataSource,
        "download",
        lambda *a, **kw: pytest.fail("Existing incomplete native fixture is offline"),
    )
    with pytest.raises(ValueError, match="(?i)unlabeled|annotation|incomplete"):
        tiny_imagenet.DataSource()


def tiny_layout(root, native):
    image_file(root / "train/n001/images/train.png")
    if native:
        image_file(root / "val/images/evaluate.png")
        (root / "val/val_annotations.txt").write_text("evaluate.png\tn001\n")
        return root / "val/images/evaluate.png"
    image_file(root / "test/n001/evaluate.png")
    return root / "test/n001/evaluate.png"


@pytest.mark.parametrize("native", [False, True])
def test_tiny_default_evaluation_is_deterministic_and_training_stays_augmented(
    temp_config, tmp_path, native
):
    Config.params["data_path"] = str(tmp_path)
    target = tiny_layout(tmp_path, native)
    rows, cols = np.indices((64, 64))
    pixels = np.stack((rows * 4, cols * 4, (rows + cols) * 2), axis=2).astype(np.uint8)
    Image.fromarray(pixels).save(target)
    Image.fromarray(pixels).save(tmp_path / "train/n001/images/train.png")
    source = tiny_imagenet.DataSource()
    torch.manual_seed(27)
    first, second = source.testset[0][0], source.testset[0][0]
    assert torch.equal(first, second)
    assert first.shape == (3, 299, 299)
    assert torch.isfinite(first).all()
    assert not torch.equal(source.trainset[0][0], source.trainset[0][0])


@pytest.mark.parametrize("native", [False, True])
def test_tiny_default_evaluation_normalizes_known_constant_pixels(
    temp_config, tmp_path, native
):
    Config.params["data_path"] = str(tmp_path)
    tiny_layout(tmp_path, native)
    source = tiny_imagenet.DataSource()
    tensor = source.testset[0][0]
    expected = torch.tensor(
        [
            (10 / 255 - 0.485) / 0.229,
            (20 / 255 - 0.456) / 0.224,
            (30 / 255 - 0.406) / 0.225,
        ]
    )[:, None, None].expand(3, 299, 299)
    torch.testing.assert_close(tensor, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("native", [False, True])
def test_tiny_explicit_train_and_test_transforms_remain_independent(
    temp_config, tmp_path, native
):
    Config.params["data_path"] = str(tmp_path)
    tiny_layout(tmp_path, native)
    source = tiny_imagenet.DataSource(
        train_transform=lambda image: "training",
        test_transform=lambda image: "evaluation",
    )
    assert source.trainset[0] == ("training", 0)
    assert source.testset[0] == ("evaluation", 0)


@pytest.mark.parametrize(
    "module,use_default_url",
    [(cinic10, False), (tiny_imagenet, False), (tiny_imagenet, True)],
)
def test_downloaded_image_dataset_recovers_stale_completion_marker(
    temp_config, tmp_path, monkeypatch, module, use_default_url
):
    Config.params["data_path"] = str(tmp_path)
    image = io.BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image, format="PNG")
    archive = io.BytesIO()
    name = "cinic" if module is cinic10 else "tiny"
    with zipfile.ZipFile(archive, "w") as zipped:
        if module is cinic10:
            paths = ["train/n001/one.png", "test/n001/evaluate.png"]
        else:
            paths = [
                "tiny-imagenet-200/train/n001/images/one.png",
                "tiny-imagenet-200/val/images/evaluate.png",
            ]
            zipped.writestr(
                "tiny-imagenet-200/val/val_annotations.txt", "evaluate.png\tn001\n"
            )
        for path in paths:
            zipped.writestr(path, image.getvalue())
    with serve(archive.getvalue(), name + ".zip") as url:
        if use_default_url:
            actual_get = module.base.requests.get

            def redirected_get(requested_url, **kwargs):
                assert requested_url == (
                    "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
                )
                return actual_get(url, **kwargs)

            monkeypatch.setattr(module.base.requests, "get", redirected_get)
        else:
            Config.data.download_url = url
        initial = module.DataSource()
        assert initial.num_test_examples() == 1
        marker = tmp_path / (
            "tiny-imagenet-200.zip.complete"
            if use_default_url
            else name + ".zip.complete"
        )
        assert marker.is_file()
        if module is cinic10:
            required = tmp_path / "test"
            shutil.rmtree(required)
        else:
            required = tmp_path / "tiny-imagenet-200/val/val_annotations.txt"
            required.unlink()
        recovered = module.DataSource()
        assert required.exists()
        assert recovered.num_test_examples() == 1
        assert recovered.testset.targets == [0]
        assert marker.is_file()
    # The server is now closed: a valid completed root must still work offline.
    assert module.DataSource().num_test_examples() == 1


def test_download_missing_required_artifacts_never_marks_complete(tmp_path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("unrelated.txt", "no dataset here")
    with serve(archive.getvalue(), "incomplete.zip") as url:
        with pytest.raises(RuntimeError, match="required.*artifacts"):
            base.DataSource.download(
                url, str(tmp_path), ready=lambda: (tmp_path / "train").is_dir()
            )
    assert not (tmp_path / "incomplete.zip.complete").exists()
