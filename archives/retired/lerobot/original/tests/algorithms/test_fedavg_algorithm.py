"""Tests for FedAvg payload filtering and dtype-safe weight handling."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from plato.algorithms.fedavg import Algorithm as FedAvgAlgorithm


class AdapterToyModel(torch.nn.Module):
    """Toy model exposing adapter-mode metadata used by SmolVLA integration."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.adapter = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("token_count", torch.tensor([7], dtype=torch.int64))
        self.plato_finetune_mode = "adapter"
        self.plato_trainable_parameter_names = ("adapter.weight",)


class DtypeToyModel(torch.nn.Module):
    """Toy model with mixed dtypes for casting safeguards."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32))
        self.register_buffer("step", torch.tensor([1], dtype=torch.int64))
        self.register_buffer("flag", torch.tensor([True, False], dtype=torch.bool))


class BFloat16ToyModel(torch.nn.Module):
    """Toy model for bf16 transport-cast regression coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones((2, 2), dtype=torch.bfloat16)
        )


def _algorithm_for(model: torch.nn.Module) -> FedAvgAlgorithm:
    trainer = cast(Any, SimpleNamespace(model=model))
    return FedAvgAlgorithm(trainer=trainer)


def _clone_state_dict(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().clone()) for name, tensor in model.state_dict().items()
    )


def test_adapter_payload_extract_load_round_trip():
    """Adapter mode should exchange only trainable tensors and load them safely."""
    torch.manual_seed(1)
    source_model = AdapterToyModel()
    source_algorithm = _algorithm_for(source_model)

    payload = source_algorithm.extract_weights()
    assert list(payload.keys()) == ["adapter.weight"]

    torch.manual_seed(2)
    target_model = AdapterToyModel()
    before = _clone_state_dict(target_model)

    target_algorithm = _algorithm_for(target_model)
    target_algorithm.load_weights(payload)

    after = target_model.state_dict()
    assert torch.equal(after["adapter.weight"], payload["adapter.weight"])
    assert torch.equal(after["backbone.weight"], before["backbone.weight"])
    assert torch.equal(after["backbone.bias"], before["backbone.bias"])
    assert torch.equal(after["token_count"], before["token_count"])

    round_trip_payload = target_algorithm.extract_weights()
    assert list(round_trip_payload.keys()) == ["adapter.weight"]
    assert torch.equal(round_trip_payload["adapter.weight"], payload["adapter.weight"])


def test_load_weights_casts_dtype_and_rounds_non_float_tensors():
    """Incoming partial payloads should be cast to model dtypes."""
    model = DtypeToyModel()
    algorithm = _algorithm_for(model)

    inbound = OrderedDict(
        {
            "weight": torch.full((2, 2), 2.5, dtype=torch.float64),
            "step": torch.tensor([3.6], dtype=torch.float32),
            "flag": torch.tensor([0.2, 0.8], dtype=torch.float32),
        }
    )

    algorithm.load_weights(inbound)

    state = model.state_dict()
    assert state["weight"].dtype == torch.float32
    assert torch.allclose(state["weight"], torch.full_like(state["weight"], 2.5))
    assert state["step"].dtype == torch.int64
    assert int(state["step"].item()) == 4
    assert state["flag"].dtype == torch.bool
    assert torch.equal(state["flag"], torch.tensor([False, True]))


def test_extract_weights_casts_bfloat16_payloads_for_transport():
    """bf16 tensors should be cast to fp32 for safe payload serialization."""
    model = BFloat16ToyModel()
    algorithm = _algorithm_for(model)

    payload = algorithm.extract_weights()
    assert payload["weight"].dtype == torch.float32

    inbound = OrderedDict(
        {"weight": torch.full((2, 2), 3.5, dtype=torch.float32)}
    )
    algorithm.load_weights(inbound)

    state = model.state_dict()
    assert state["weight"].dtype == torch.bfloat16
    assert torch.allclose(
        state["weight"].float(),
        torch.full((2, 2), 3.5, dtype=torch.float32),
    )


def test_extract_weights_respects_optional_payload_size_limit():
    """Payload extraction should fail fast if a configured max size is exceeded."""
    model = torch.nn.Linear(32, 32, bias=False)
    setattr(model, "plato_max_payload_size_mb", 0.0001)
    algorithm = _algorithm_for(model)

    with pytest.raises(ValueError, match="payload size"):
        algorithm.extract_weights()


def test_fedavg_full_mode_round_trip_with_large_weights():
    """Full-state FedAvg flow should remain compatible for non-adapter models."""
    torch.manual_seed(3)
    model = torch.nn.Linear(1024, 1024, bias=False)
    algorithm = _algorithm_for(model)

    baseline = algorithm.extract_weights()
    assert list(baseline.keys()) == ["weight"]

    received = [OrderedDict((name, tensor + 0.25) for name, tensor in baseline.items())]
    deltas = algorithm.compute_weight_deltas(baseline, received)
    updated = algorithm.update_weights(deltas[0])

    algorithm.load_weights(updated)

    state = model.state_dict()
    assert torch.allclose(state["weight"], received[0]["weight"])
