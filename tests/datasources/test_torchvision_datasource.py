"""Tests for the generic torchvision datasource."""

from __future__ import annotations

import types

from plato.config import Config as BaseConfig
from plato.datasources import torchvision as torchvision_ds


class _StubTransforms:
    """Minimal replacement for torchvision.transforms."""

    def ToTensor(self):
        return "to_tensor"


def _build_config(tmp_path, data_dict):
    """Construct a stub Config object exposing the expected attributes."""
    data_node = BaseConfig.node_from_dict(data_dict)
    params = {"data_path": str(tmp_path)}
    return types.SimpleNamespace(data=data_node, params=params)


def test_torchvision_datasource_supports_named_splits(monkeypatch, tmp_path):
    """Datasets exposing a `split` argument should map to the requested subsets."""

    class DummySplitDataset:
        def __init__(
            self,
            root,
            split="train",
            download=False,
            transform=None,
            target_transform=None,
        ):
            self.root = root
            self.split = split
            self.download = download
            self.transform = transform
            self.target_transform = target_transform
            self.labels = [0, 1]
            self.classes = ("cat", "dog")
            self.data = [0, 1]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

    stub_datasets = types.SimpleNamespace(DummySplitDataset=DummySplitDataset)
    dummy_config = _build_config(
        tmp_path,
        {
            "datasource": "Torchvision",
            "dataset_name": "DummySplitDataset",
            "download": False,
            "unlabeled_split": "unlabeled",
        },
    )

    monkeypatch.setattr(torchvision_ds, "datasets", stub_datasets)
    monkeypatch.setattr(torchvision_ds, "transforms", _StubTransforms())
    monkeypatch.setattr(torchvision_ds, "Config", lambda: dummy_config)

    datasource = torchvision_ds.DataSource(
        train_transform="train_tx", test_transform="test_tx"
    )

    assert datasource.trainset.split == "train"
    assert datasource.trainset.transform == "train_tx"
    assert datasource.trainset.download is False
    assert datasource.trainset.targets == [0, 1]

    assert datasource.testset.split == "test"
    assert datasource.testset.transform == "test_tx"
    assert datasource.testset.download is False

    unlabeled = datasource.get_unlabeled_set()
    assert unlabeled.split == "unlabeled"
    assert unlabeled.transform == "train_tx"

    # Metadata helpers should fallback to available attributes.
    assert datasource.classes() == ["cat", "dog"]
    assert datasource.targets() == [0, 1]


def test_torchvision_datasource_supports_boolean_splits_and_kwargs(
    monkeypatch, tmp_path
):
    """Datasets using boolean `train` splits should receive defaults and overrides."""

    class DummyBoolDataset:
        def __init__(
            self,
            root,
            train=True,
            download=False,
            transform=None,
            sample_rate=1.0,
        ):
            self.root = root
            self.train = train
            self.download = download
            self.transform = transform
            self.sample_rate = sample_rate
            self.targets = [int(train)] * 3
            self.classes = ("neg", "pos")

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return index, self.targets[index]

    stub_datasets = types.SimpleNamespace(DummyBoolDataset=DummyBoolDataset)
    dummy_config = _build_config(
        tmp_path,
        {
            "datasource": "Torchvision",
            "dataset_name": "DummyBoolDataset",
            "download": True,
            "train_kwargs": {"sample_rate": 0.5},
            "test_kwargs": {"sample_rate": 0.25},
        },
    )

    monkeypatch.setattr(torchvision_ds, "datasets", stub_datasets)
    monkeypatch.setattr(torchvision_ds, "transforms", _StubTransforms())
    monkeypatch.setattr(torchvision_ds, "Config", lambda: dummy_config)

    datasource = torchvision_ds.DataSource()

    assert datasource.trainset.train is True
    assert datasource.trainset.download is True
    assert datasource.trainset.sample_rate == 0.5

    assert datasource.testset.train is False
    assert datasource.testset.download is False
    assert datasource.testset.sample_rate == 0.25
    assert datasource.testset.transform == "to_tensor"

    assert datasource.targets() == [1, 1, 1]
    assert datasource.classes() == ["neg", "pos"]
