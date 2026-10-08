"""Real local dataset fixtures: cache recovery, counts, labels and seed isolation."""

import json
import struct

import numpy as np
import pytest
from PIL import Image

from plato.config import Config
from plato.datasources import cinic10, femnist, purchase, registry, texas, tiny_imagenet


@pytest.mark.parametrize("module", [purchase, texas])
def test_vector_dataset_small_cache_has_real_counts_and_preserves_rng(
    temp_config, tmp_path, monkeypatch, module
):
    Config.params["data_path"] = str(tmp_path)
    cache_name = "purchase_numpy.npz" if module is purchase else "texas_numpy.npz"
    features = np.array([[i, i + 0.5] for i in range(7)])
    labels = np.arange(7)
    np.savez(tmp_path / cache_name, X=features, Y=labels)

    def unexpected_download(*args, **kwargs):
        pytest.fail("A complete NumPy cache must not trigger a download")

    monkeypatch.setattr(module.request, "urlretrieve", unexpected_download)
    before = np.random.get_state()
    source = module.DataSource()
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert source.num_train_examples() == len(source.trainset) == 7
    assert source.num_test_examples() == len(source.testset) == 0
    assert sorted(source.targets()) == list(range(7))
    assert all(sample[0].item() == label.item() for sample, label in source.trainset)


@pytest.mark.parametrize("module", [purchase, texas])
def test_existing_raw_vector_data_is_processed_without_download(
    temp_config, tmp_path, monkeypatch, module
):
    Config.params["data_path"] = str(tmp_path)

    def unexpected_download(*args, **kwargs):
        pytest.fail("Existing complete raw dataset should be processed locally")

    monkeypatch.setattr(module.request, "urlretrieve", unexpected_download)
    if module is purchase:
        (tmp_path / "dataset_purchase").write_text("1,0.1,0.2\n2,0.3,0.4\n")
    else:
        folder = tmp_path / "texas/100"
        folder.mkdir(parents=True)
        (folder / "feats").write_text("0.1,0.2\n0.3,0.4\n")
        (folder / "labels").write_text("1\n2\n")
    source = module.DataSource()
    assert len(source.trainset) == 2
    assert sorted(source.targets()) == [0, 1]


@pytest.mark.parametrize("module", [purchase, texas])
def test_vector_splits_preserve_legacy_seed_order_without_global_mutation(
    temp_config, tmp_path, module
):
    Config.params["data_path"] = str(tmp_path)
    name = "purchase" if module is purchase else "texas"
    rows = np.arange(40001)
    np.savez(tmp_path / (name + "_numpy.npz"), X=rows.reshape(-1, 1), Y=rows % 100)
    before = np.random.get_state()
    source = module.DataSource()
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert source.num_train_examples() == source.num_test_examples() == 20000
    expected_rows = [12836, 10913, 4214, 8198, 31403]
    assert [int(source.trainset[i][0].item()) for i in range(5)] == expected_rows
    assert [int(source.trainset[i][1].item()) for i in range(5)] == [
        row % 100 for row in expected_rows
    ]


def image_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)


def test_cinic_existing_empty_root_recovers_download(
    temp_config, tmp_path, monkeypatch
):
    Config.params["data_path"] = str(tmp_path)
    Config.data.download_url = "https://fixture.invalid/cinic.tar.gz"
    calls = []

    def download(url, data_path, **kwargs):
        calls.append(url)
        for split in ("train", "test"):
            image_file(tmp_path / split / "class0/image.png")

    monkeypatch.setattr(cinic10.DataSource, "download", download)
    source = cinic10.DataSource()
    assert calls == [Config.data.download_url]
    assert source.num_train_examples() == source.num_test_examples() == 1
    assert source.classes() == ["class0"]


@pytest.mark.parametrize("preexisting_native", [False, True])
def test_tiny_imagenet_canonical_validation_labels(
    temp_config, tmp_path, monkeypatch, preexisting_native
):
    Config.params["data_path"] = str(tmp_path)
    Config.data.download_url = "https://fixture.invalid/tiny.zip"

    def download(url, data_path, **kwargs):
        root = tmp_path if preexisting_native else tmp_path / "tiny-imagenet-200"
        image_file(root / "train/n001/images/train.png")
        image_file(root / "train/n002/images/train.png")
        image_file(root / "val/images/first.png")
        image_file(root / "val/images/second.png")
        image_file(root / "test/images/unlabeled.png")
        (root / "val/val_annotations.txt").write_text(
            "first.png\tn002\t0\t0\t8\t8\nsecond.png\tn001\t0\t0\t8\t8\n"
        )

    if preexisting_native:
        download(None, None)

        def unexpected_download(*args, **kwargs):
            pytest.fail("A complete native Tiny ImageNet tree must be reused")

        monkeypatch.setattr(tiny_imagenet.DataSource, "download", unexpected_download)
    else:
        monkeypatch.setattr(tiny_imagenet.DataSource, "download", download)
    source = tiny_imagenet.DataSource()
    assert source.trainset is not None
    assert source.testset is not None
    assert source.num_train_examples() == source.num_test_examples() == 2
    assert source.classes() == ["n001", "n002"]
    assert source.testset is not None
    assert source.testset.targets == [1, 0]
    assert [label for _, label in source.testset] == [1, 0]
    assert source.testset[0][0].shape[0] == 3


def test_tiny_imagenet_preserves_prepared_labeled_test_tree(temp_config, tmp_path):
    Config.params["data_path"] = str(tmp_path)
    image_file(tmp_path / "train/n001/one.png")
    image_file(tmp_path / "test/n001/one.png")
    source = tiny_imagenet.DataSource()
    assert source.trainset is not None
    assert source.testset is not None
    assert source.num_train_examples() == source.num_test_examples() == 1
    assert source.classes() == ["n001"]
    assert source.testset[0][1] == 0


def test_prepartitioned_femnist_ids_and_counts(temp_config, tmp_path):
    Config.params["data_path"] = str(tmp_path)
    for client_id, split, count in [(0, "test", 3), (1, "train", 2), (2, "train", 4)]:
        folder = tmp_path / "FEMNIST/packaged_data" / split / str(client_id)
        folder.mkdir(parents=True)
        (folder / "data.json").write_text(
            json.dumps(
                {"x": [[0.0] * 784 for _ in range(count)], "y": list(range(count))}
            )
        )
        source = femnist.DataSource(client_id=client_id, train_transform=lambda x: x)
        assert source.trainset is not None
        assert source.testset is not None
        assert source.num_train_examples() == source.num_test_examples() == count
        assert source.trainset[0][1] == 0


def test_torchvision_alias_selection_with_real_local_idx_files(temp_config, tmp_path):
    Config.params["data_path"] = str(tmp_path)
    Config.data.download = False
    for name, labels in [("MNIST", [0, 1, 2]), ("FashionMNIST", [7, 8, 9])]:
        raw = tmp_path / name / "raw"
        raw.mkdir(parents=True)
        for images, targets in [
            ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        ]:
            (raw / images).write_bytes(
                struct.pack(">IIII", 2051, 3, 28, 28) + bytes(3 * 784)
            )
            (raw / targets).write_bytes(struct.pack(">II", 2049, 3) + bytes(labels))
    mnist = registry.get(datasource_name="MNIST")
    fashion = registry.get(datasource_name="FashionMNIST")
    assert mnist.targets() == [0, 1, 2]
    assert fashion.targets() == [7, 8, 9]
    assert fashion.trainset[0][0].shape == (1, 28, 28)
    assert mnist.num_train_examples() == fashion.num_test_examples() == 3


@pytest.mark.parametrize("variant", [None, "digits"])
def test_emnist_variant_keeps_distinct_training_and_test_data(
    temp_config, tmp_path, variant
):
    Config.params["data_path"] = str(tmp_path)
    Config.data.download = False
    if variant is not None:
        Config.data.dataset_kwargs = {"split": variant}
    selected_variant = variant or "balanced"
    raw = tmp_path / "EMNIST/raw"
    raw.mkdir(parents=True)
    for split, labels in [("train", [0, 1, 2]), ("test", [7, 8])]:
        prefix = f"emnist-{selected_variant}-{split}"
        (raw / (prefix + "-images-idx3-ubyte")).write_bytes(
            struct.pack(">IIII", 2051, len(labels), 28, 28) + bytes(len(labels) * 784)
        )
        (raw / (prefix + "-labels-idx1-ubyte")).write_bytes(
            struct.pack(">II", 2049, len(labels)) + bytes(labels)
        )

    source = registry.get(datasource_name="EMNIST")
    assert source.trainset.targets.tolist() == [0, 1, 2]
    assert source.testset.targets.tolist() == [7, 8]
    assert source.num_train_examples() == 3
    assert source.num_test_examples() == 2
    assert source.trainset.train is True
    assert source.testset.train is False
    assert source.trainset.split == source.testset.split == selected_variant
    assert source.testset[0][0].shape == (1, 28, 28)
