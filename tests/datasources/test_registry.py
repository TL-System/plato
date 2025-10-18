"""Tests covering datasource registry selection and fallbacks."""

from types import SimpleNamespace

import pytest
import torch

from plato.datasources import base, registry


class _FakeDataset:
    """Minimal dataset exposing targets/classes for base.DataSource helpers."""

    def __init__(self, length: int, shape):
        self.samples = [torch.zeros(shape) for _ in range(length)]
        self.targets = list(range(length))
        self.classes = ("class0", "class1")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.targets[index % len(self.classes)]


class _VisionStubModule:
    """Module-style container emulating torchvision-backed datasources."""

    class DataSource(base.DataSource):
        def __init__(self, **unused_kwargs):
            super().__init__()
            self.trainset = _FakeDataset(length=5, shape=(1, 28, 28))
            self.testset = _FakeDataset(length=2, shape=(1, 28, 28))


def test_registry_returns_stub_datasource(monkeypatch):
    """Registry should instantiate registered modules and expose metadata helpers."""
    monkeypatch.setitem(
        registry.registered_datasources, "FakeVision", _VisionStubModule
    )

    datasource = registry.get(datasource_name="FakeVision")

    assert isinstance(datasource, _VisionStubModule.DataSource)
    assert datasource.num_train_examples() == 5
    assert datasource.num_test_examples() == 2
    assert datasource.classes() == ["class0", "class1"]


def test_registry_uses_configured_name(monkeypatch):
    """Config-sourced datasource names should be honoured when kwargs are absent."""
    monkeypatch.setitem(
        registry.registered_datasources, "FakeVision", _VisionStubModule
    )

    class DummyConfig:
        def __init__(self):
            self.data = SimpleNamespace(datasource="FakeVision")

    monkeypatch.setattr(registry, "Config", lambda: DummyConfig())

    datasource = registry.get()

    assert isinstance(datasource, _VisionStubModule.DataSource)


def test_registry_supports_feature_datasource(monkeypatch):
    """Feature datasource should flatten batched tensors into per-sample items."""

    class DummyConfig:
        def __init__(self):
            self.data = SimpleNamespace(datasource="Feature")

    monkeypatch.setattr(registry, "Config", lambda: DummyConfig())

    batch_features = torch.randn(3, 2)
    batch_targets = torch.tensor([1, 0, 1])
    datasource = registry.get(
        features=[[[(batch_features, batch_targets)]]], datasource_name="Feature"
    )

    assert len(datasource) == 3
    feature, label = datasource[0]
    assert feature.shape == (2,)
    assert label.item() in {0, 1}
    assert datasource.testset == []


def test_registry_raises_for_unknown_datasource(monkeypatch):
    """Missing datasources should raise a helpful ValueError."""

    class DummyConfig:
        def __init__(self):
            self.data = SimpleNamespace(datasource="MissingSet")

    monkeypatch.setattr(registry, "Config", lambda: DummyConfig())

    with pytest.raises(ValueError):
        registry.get()
