"""Actual callback and public end-hook rejection boundaries for SCAFFOLD."""

import asyncio
import copy
import json
import pickle
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from plato.callbacks.server import ServerCallback
from plato.serialization.safetensor import serialize_tree
from plato.trainers.strategies.training_step import GradientAccumulationStepStrategy
from tests.integration.test_scaffold_round_flow import (
    create_client,
    local_round,
    make_server,
    shipped_config,
)
from tests.integration.utils import configure_environment
from tests.trainers.test_scaffold_strategy import scalar_controls


class InterruptBatch(ServerCallback):
    def __init__(self, mode):
        self.mode = mode

    def on_weights_received(self, server, weights):
        if self.mode == "receive":
            raise RuntimeError("Deliberate receive callback failure")
        if self.mode == "metadata":
            server.updates[-1].report.num_samples = -1
        if self.mode == "model":
            weights[-1]["theta"].fill_(float("nan"))
        if self.mode == "pending":
            server._pending_control_variate["theta"].fill_(float("nan"))
        if self.mode == "delta":
            server.received_client_control_variates[-1]["theta"].fill_(float("nan"))
        if self.mode == "committed-control":
            server.server_control_variate["theta"].fill_(float("nan"))

    def on_weights_aggregated(self, server, updates):
        if self.mode == "aggregate":
            raise RuntimeError("Deliberate aggregate callback failure")
        if self.mode == "aggregate-model":
            server.trainer.model.theta.data.fill_(float("nan"))
        if self.mode == "aggregate-control":
            server.server_control_variate["theta"].add_(1)

    def on_clients_processed(self, server):
        if self.mode == "postcommit":
            raise RuntimeError("Deliberate postcommit reporting failure")


@pytest.mark.parametrize(
    "mode",
    [
        "receive",
        "metadata",
        "model",
        "pending",
        "delta",
        "committed-control",
        "aggregate",
        "aggregate-model",
        "aggregate-control",
        "load",
        "arithmetic",
        "postcommit",
    ],
)
def test_actual_report_callbacks_reject_and_clear_before_successful_retry(
    tmp_path, monkeypatch, mode
):
    with configure_environment(shipped_config(), runtime_root=tmp_path):
        server = make_server()
        server.server_control_variate = scalar_controls(2.0)
        client = create_client(1)
        client.trainer.model_update_strategy.client_control_variate = scalar_controls(1)
        report, _ = local_round(
            client,
            [scalar_controls(1), scalar_controls(2)],
            round_id=1,
            target=0.0,
            samples=2,
        )
        server.updates = [
            SimpleNamespace(
                report=copy.copy(report),
                payload=[scalar_controls(y), scalar_controls(delta)],
            )
            for y, delta in ((7, 3), (9, 4))
        ]
        raw = [serialize_tree(u.payload) for u in server.updates]
        callback = InterruptBatch(mode)
        server.callback_handler.add_callback(callback)
        original_loader = server.algorithm.load_weights
        original_aggregation = server.aggregation_strategy.aggregate_weights
        if mode == "load":

            def partial_load(weights):
                original_loader(weights)
                raise RuntimeError("Deliberate partially loaded model failure")

            monkeypatch.setattr(server.algorithm, "load_weights", partial_load)
        if mode == "arithmetic":

            async def fail_aggregation(*args, **kwargs):
                raise RuntimeError("Deliberate aggregation failure")

            monkeypatch.setattr(
                server.aggregation_strategy, "aggregate_weights", fail_aggregation
            )
        with pytest.raises((RuntimeError, ValueError)):
            asyncio.run(server._process_reports())
        expected_x, expected_c = (8, 13 / 3) if mode == "postcommit" else (1, 2)
        assert server.trainer.model.theta.item() == pytest.approx(expected_x)
        assert server.server_control_variate["theta"].item() == pytest.approx(
            expected_c
        )
        assert server._pending_control_variate is None
        assert server.received_client_control_variates is None
        assert raw == [serialize_tree(u.payload) for u in server.updates]
        # A later ingress error must not resurrect the rejected stage.
        server.updates[-1].report.num_samples = -1
        with pytest.raises(ValueError, match="sample"):
            asyncio.run(server._process_reports())
        assert server._pending_control_variate is None
        assert server.received_client_control_variates is None
        server.callback_handler.callbacks.remove(callback)
        monkeypatch.setattr(server.algorithm, "load_weights", original_loader)
        monkeypatch.setattr(
            server.aggregation_strategy, "aggregate_weights", original_aggregation
        )
        for update in server.updates:
            update.report.num_samples = 2
        asyncio.run(server._process_reports())
        assert server.trainer.model.theta.item() == 8
        assert server.server_control_variate["theta"].item() == pytest.approx(
            expected_c + 7 / 3
        )
        assert server._pending_control_variate is None
        assert server.received_client_control_variates is None


class InterruptEnd(GradientAccumulationStepStrategy):
    def __init__(self):
        super().__init__(1)
        self.end_calls = 0

    def on_train_end(self, context):
        self.end_calls += 1
        super().on_train_end(context)
        raise RuntimeError("Deliberate training step end failure")


@pytest.mark.parametrize("interface", ["train_model", "train_process"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("cleanup_failure", [False, True])
@pytest.mark.parametrize("end_failure", [False, True])
def test_fallible_end_hook_rejects_before_acceptance_and_releases_ownership(
    tmp_path, monkeypatch, interface, existing, cleanup_failure, end_failure
):
    config = shipped_config()
    with configure_environment(config, runtime_root=tmp_path):
        client = create_client(1)
        trainer, strategy = client.trainer, client.trainer.model_update_strategy
        strategy.client_control_variate = scalar_controls(1)
        trainer.context.state["server_control_variate"] = scalar_controls(2)
        canonical = Path(strategy.client_control_variate_path)
        if existing:
            canonical.write_bytes(pickle.dumps(scalar_controls(1)))
        before = canonical.read_bytes() if existing else None
        ending = trainer.training_step_strategy = InterruptEnd()
        if not end_failure:

            def successful_end(context):
                ending.end_calls += 1
                GradientAccumulationStepStrategy.on_train_end(ending, context)

            monkeypatch.setattr(ending, "on_train_end", successful_end)
        original_cleanup = strategy.on_train_cleanup
        cleanup_calls = []

        def recording_cleanup(context, successful):
            if ending.end_calls:
                cleanup_calls.append(successful)
            original_cleanup(context, successful)
            if cleanup_failure and ending.end_calls:
                raise RuntimeError("Deliberate secondary cleanup failure")

        monkeypatch.setattr(strategy, "on_train_cleanup", recording_cleanup)
        data = TensorDataset(torch.ones(2, 1).double(), torch.zeros(2, 1).double())
        if not cleanup_failure and not end_failure:
            getattr(trainer, interface)(
                {**config["trainer"], "run_id": "successful-end"}, data, [0, 1]
            )
            assert ending.end_calls == 1
            assert strategy.client_control_variate["theta"].item() == pytest.approx(0.9)
            assert strategy.get_update_payload(trainer.context)[
                "control_variate_delta"
            ]["theta"].item() == pytest.approx(-0.1)
            assert not trainer.optimizer._optimizer_step_pre_hooks
            assert cleanup_calls == [True]
            return
        with pytest.raises(
            (RuntimeError, BaseExceptionGroup),
            match=("cleanup" if cleanup_failure else "training step end"),
        ) as failure:
            getattr(trainer, interface)(
                {**config["trainer"], "run_id": "end-failure"}, data, [0, 1]
            )
        assert ending.end_calls == 1
        assert cleanup_calls == ([False] if end_failure else [True, False])
        if end_failure and cleanup_failure:
            assert isinstance(failure.value, BaseExceptionGroup)
            assert "training step end failure" in str(failure.value.exceptions[0])
            assert "secondary cleanup failure" in str(failure.value.exceptions[1])
        assert strategy.client_control_variate["theta"].item() == 1
        assert canonical.exists() is existing
        if existing:
            assert canonical.read_bytes() == before
        with pytest.raises(RuntimeError, match="successful"):
            strategy.get_update_payload(trainer.context)
        assert not trainer.optimizer._optimizer_step_pre_hooks
        assert "complete_optimizer_step" not in trainer.context.state
        assert "optimizer_step_hooks_handled" not in trainer.context.state
        monkeypatch.setattr(strategy, "on_train_cleanup", original_cleanup)
        for client_id in (2, 1):
            client.client_id = client._context.client_id = client_id
            client.configure()
        reconstructed = create_client(1).trainer.model_update_strategy
        if existing:
            assert reconstructed.client_control_variate["theta"].item() == 1
        else:
            assert reconstructed.client_control_variate is None
            # No persisted state was accepted; explicitly retain the caller's
            # original ci=1 for the numerical retry, rather than adopting zero.
            strategy.client_control_variate = scalar_controls(1)
        trainer.training_step_strategy = GradientAccumulationStepStrategy(1)
        _, retry = local_round(
            client,
            [scalar_controls(1), scalar_controls(2)],
            round_id=2,
            target=0.0,
            samples=2,
        )
        assert trainer.model.theta.item() == pytest.approx(0.62, abs=1e-12)
        assert strategy.client_control_variate["theta"].item() == pytest.approx(
            0.9, abs=1e-12
        )
        assert retry[1]["theta"].item() == pytest.approx(-0.1, abs=1e-12)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.slow
def test_actual_spawned_successful_run_cleanup_rejects_before_parent_acceptance(
    tmp_path, existing
):
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        "-m",
        "tests.integration.scaffold_cleanup_worker",
        str(tmp_path / "runtime"),
        str(output),
        "1" if existing else "0",
    ]
    completed = subprocess.run(
        ["zsh", "-lc", shlex.join(command)], capture_output=True, text=True, timeout=90
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text())
    assert result == pytest.approx(dict(y=0.62, ci=0.9, delta=-0.1), abs=1e-12)
