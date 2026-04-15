"""Tests for the FedDF example aggregation strategy."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from plato.config import Config

_TESTS_ROOT = Path(__file__).resolve().parent
_FEDDF_DIR = (
    _TESTS_ROOT.parent.parent / "examples" / "server_aggregation" / "feddf"
)
if str(_FEDDF_DIR) not in sys.path:
    sys.path.insert(0, str(_FEDDF_DIR))


def _load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise RuntimeError(f"Unable to load spec for {path}")

    module = cast(Any, importlib.util.module_from_spec(spec))
    loader = spec.loader
    if loader is None:
        raise RuntimeError(f"Loader missing for {path}")

    loader.exec_module(module)
    return module


feddf_algorithm = _load_module(
    "feddf_algorithm_module",
    _FEDDF_DIR / "feddf_algorithm.py",
)
feddf_server_strategy = _load_module(
    "feddf_server_strategy_module",
    _FEDDF_DIR / "feddf_server_strategy.py",
)


class TinyStudent(torch.nn.Module):
    """Small student model used to verify server-side distillation."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def test_feddf_server_process_reports_distills_global_model(temp_config):
    """FedDF should consume logits payloads and update the global model."""
    from plato.servers import fedavg

    Config().server.do_test = False

    trainer = SimpleNamespace(model=TinyStudent(), device="cpu")
    algorithm = feddf_algorithm.Algorithm(trainer=trainer)
    strategy = feddf_server_strategy.FedDFAggregationStrategy(
        temperature=1.0,
        distillation_epochs=80,
        distillation_batch_size=2,
        distillation_learning_rate=0.4,
    )
    server = fedavg.Server(aggregation_strategy=strategy)
    server.algorithm = algorithm
    server.trainer = trainer
    server.context.server = server
    server.context.algorithm = algorithm
    server.context.trainer = trainer
    server.context.state["prng_state"] = None

    proxy_inputs = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.5, 0.2],
            [0.2, 1.5],
        ]
    )
    proxy_dataset = TensorDataset(proxy_inputs, torch.zeros(len(proxy_inputs)))
    server.context.state["feddf_proxy_dataset"] = proxy_dataset

    teacher_logits_a = torch.tensor(
        [
            [7.0, -7.0],
            [-7.0, 7.0],
            [5.5, -5.5],
            [-5.5, 5.5],
        ]
    )
    teacher_logits_b = torch.tensor(
        [
            [6.0, -6.0],
            [-6.0, 6.0],
            [4.5, -4.5],
            [-4.5, 4.5],
        ]
    )

    server.updates = [
        SimpleNamespace(
            client_id=1,
            report=SimpleNamespace(
                num_samples=1,
                accuracy=0.0,
                processing_time=0.0,
                comm_time=0.0,
                training_time=0.0,
            ),
            payload={"logits": teacher_logits_a},
        ),
        SimpleNamespace(
            client_id=2,
            report=SimpleNamespace(
                num_samples=1,
                accuracy=0.0,
                processing_time=0.0,
                comm_time=0.0,
                training_time=0.0,
            ),
            payload={"logits": teacher_logits_b},
        ),
    ]

    baseline_log_probs = torch.log_softmax(torch.zeros_like(teacher_logits_a), dim=1)
    teacher_targets = torch.softmax((teacher_logits_a + teacher_logits_b) / 2, dim=1)
    baseline_loss = F.kl_div(
        baseline_log_probs,
        teacher_targets,
        reduction="batchmean",
    )

    asyncio.run(server._process_reports())

    updated_weights = algorithm.extract_weights()
    assert not torch.allclose(
        updated_weights["linear.weight"],
        torch.zeros_like(updated_weights["linear.weight"]),
    )

    with torch.no_grad():
        updated_logits = trainer.model(proxy_inputs)
        updated_log_probs = torch.log_softmax(updated_logits, dim=1)

    distilled_loss = F.kl_div(
        updated_log_probs,
        teacher_targets,
        reduction="batchmean",
    )
    assert distilled_loss < baseline_loss
