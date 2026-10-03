"""Tests for SmolVLA model registry construction and adapter metadata."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from plato.algorithms.fedavg import Algorithm as FedAvgAlgorithm
from plato.models import registry as models_registry
from plato.models import smolvla as smolvla_model
from tests.test_utils.lerobot_stubs import FakeSmolVLAPolicy


def test_model_registry_constructs_smolvla_wrapper(temp_config, monkeypatch):
    """Model registry should construct SmolVLA via the registered factory."""
    FakeSmolVLAPolicy.reset_calls()
    monkeypatch.setattr(
        smolvla_model,
        "_import_smolvla_policy",
        lambda: FakeSmolVLAPolicy,
    )

    model = models_registry.get(
        model_name="smolvla",
        model_type="smolvla",
        model_params={
            "path": "stub/smolvla",
            "finetune_mode": "adapter",
            "adapter_parameter_patterns": ["adapter"],
            "strict": True,
        },
    )

    assert isinstance(model, FakeSmolVLAPolicy)
    assert model.plato_policy_path == "stub/smolvla"
    assert model.plato_finetune_mode == "adapter"
    assert model.plato_trainable_parameter_names == ("adapter.weight",)
    assert FakeSmolVLAPolicy.load_calls[-1]["path"] == "stub/smolvla"


def test_smolvla_adapter_metadata_filters_fedavg_payload(
    temp_config,
    monkeypatch,
):
    """Regression: adapter mode metadata must drive adapter-only FedAvg payloads."""
    FakeSmolVLAPolicy.reset_calls()
    monkeypatch.setattr(
        smolvla_model,
        "_import_smolvla_policy",
        lambda: FakeSmolVLAPolicy,
    )

    model = smolvla_model.Model.get(
        policy_path="stub/smolvla",
        finetune_mode="adapter",
        adapter_parameter_patterns=["adapter"],
    )

    trainer = cast(Any, SimpleNamespace(model=model))
    algorithm = FedAvgAlgorithm(trainer=trainer)
    payload = algorithm.extract_weights()

    assert list(payload.keys()) == ["adapter.weight"]
    assert "backbone.weight" not in payload
