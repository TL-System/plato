"""Tests for the model registry helpers."""

from types import SimpleNamespace

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
