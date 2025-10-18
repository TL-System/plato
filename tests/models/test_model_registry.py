"""Tests for the model registry helpers."""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from plato.models import registry as models_registry


def test_model_registry_instantiates_configured_model(monkeypatch):
    """The registry should build the model described in the configuration."""

    dummy_config = SimpleNamespace(
        trainer=SimpleNamespace(model_name="lenet5", model_type="lenet5"),
        parameters=SimpleNamespace(model=SimpleNamespace(_asdict=lambda: {})),
    )

    monkeypatch.setattr(models_registry, "Config", lambda: dummy_config)

    model = models_registry.get()

    assert isinstance(model, nn.Module)
    assert hasattr(model, "forward")


def test_model_registry_passes_kwargs_to_registered_model(temp_config, monkeypatch):
    """Explicit kwargs should be forwarded into the registered model constructor."""

    captured = {}

    class DummyModel(nn.Module):
        def __init__(self, alpha, **unused_kwargs):
            super().__init__()
            captured["alpha"] = alpha

    monkeypatch.setitem(models_registry.registered_models, "dummy", DummyModel)

    model = models_registry.get(model_type="dummy", model_params={"alpha": 0.3})

    assert isinstance(model, DummyModel)
    assert captured["alpha"] == 0.3


def test_model_registry_derives_type_from_model_name(temp_config, monkeypatch):
    """Model type should fall back to the prefix in the configured model name."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()

    monkeypatch.setitem(models_registry.registered_models, "derived", DummyModel)

    model = models_registry.get(model_name="derived_custom_variant")

    assert isinstance(model, DummyModel)


def test_model_registry_invokes_factory_getter(temp_config, monkeypatch):
    """Registered factories should be called via their get method."""

    sentinel = object()

    class DummyFactory:
        called_with = {}

        @staticmethod
        def get(model_name, **kwargs):
            DummyFactory.called_with = {"model_name": model_name, "kwargs": kwargs}
            return sentinel

    monkeypatch.setitem(models_registry.registered_factories, "factory", DummyFactory)

    built = models_registry.get(
        model_type="factory",
        model_name="factory_model",
        model_params={"hidden": 128},
    )

    assert built is sentinel
    assert DummyFactory.called_with == {
        "model_name": "factory_model",
        "kwargs": {"hidden": 128},
    }


def test_model_registry_raises_for_unknown_model(temp_config):
    """An informative error should be raised when the model type is missing."""

    with pytest.raises(ValueError):
        models_registry.get(model_type="missing_model")
