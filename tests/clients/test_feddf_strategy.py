"""Tests for the FedDF example training strategy."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import torch

from tests.test_utils.fakes import FakeModel

_TESTS_ROOT = Path(__file__).resolve().parent
_FEDDF_DIR = (
    _TESTS_ROOT.parent.parent / "examples" / "server_aggregation" / "feddf"
)
if str(_FEDDF_DIR) not in sys.path:
    sys.path.insert(0, str(_FEDDF_DIR))

_FEDDF_CLIENT_PATH = _FEDDF_DIR / "feddf_client.py"
_FEDDF_SPEC = importlib.util.spec_from_file_location(
    "feddf_client_module", _FEDDF_CLIENT_PATH
)
if _FEDDF_SPEC is None:
    raise RuntimeError(f"Unable to load spec for {_FEDDF_CLIENT_PATH}")

feddf_client = cast(Any, importlib.util.module_from_spec(_FEDDF_SPEC))
loader = _FEDDF_SPEC.loader
if loader is None:
    raise RuntimeError(f"Loader missing for {_FEDDF_CLIENT_PATH}")
loader.exec_module(feddf_client)


def test_feddf_training_strategy_returns_teacher_logits(temp_config):
    """FedDF clients should send proxy-set logits instead of model weights."""
    strategy = feddf_client.FedDFTrainingStrategy(
        proxy_batch_size=2,
    )
    loaded_weights = []
    context = SimpleNamespace(
        client_id=1,
        current_round=1,
        algorithm=SimpleNamespace(load_weights=lambda weights: loaded_weights.append(weights)),
        trainer=SimpleNamespace(model=FakeModel(), device="cpu"),
        state={},
    )
    proxy_inputs = torch.randn(3, 4)
    inbound_payload = {
        "weights": {"linear.weight": torch.ones(2, 4)},
        "proxy_inputs": proxy_inputs,
    }
    strategy.load_payload(context, inbound_payload)

    mock_report = SimpleNamespace(num_samples=8)
    async_mock = AsyncMock(return_value=(mock_report, {"weights": torch.ones(1)}))

    with patch.object(
        feddf_client.DefaultTrainingStrategy,
        "train",
        new=async_mock,
    ) as mock_train, patch.object(
        feddf_client.time,
        "perf_counter",
        side_effect=[10.0, 10.25],
    ):
        report, payload = asyncio.run(strategy.train(context))

    mock_train.assert_awaited_once()
    assert report is mock_report
    assert getattr(report, "training_time") == 0.25
    assert getattr(report, "feddf_proxy_logits_time") == 0.25
    assert getattr(report, "payload_type") == "feddf_logits"
    assert getattr(report, "proxy_size") == 3
    assert "logits" in payload
    assert "weights" not in payload
    assert tuple(payload["logits"].shape) == (3, 2)
    assert loaded_weights == [inbound_payload["weights"]]
    assert torch.equal(context.state["feddf_proxy_inputs"], proxy_inputs)
    assert context.state["feddf_proxy_logits_time"] == 0.25
