"""Independent count provenance and callback/checkpoint transaction boundaries."""

import asyncio
import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from plato.callbacks.server import ServerCallback
from tests.integration.test_feddyn_round_flow import (
    assert_state_equal,
    client,
    committed_state,
    configuration,
    dispatch,
    local,
    server,
)
from tests.integration.utils import configure_environment


class SingleDrawPartition:
    calls = 0

    def get(self):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("Realized partition was requested again")
        return torch.utils.data.SubsetRandomSampler([7, 13])

    def num_samples(self):
        return 2


@pytest.mark.parametrize("mode", ["uniform", "sample"])
def test_actual_spawn_wrong_first_count_rejected_and_retry(tmp_path, mode):
    output = tmp_path / "result.json"
    command = [
        sys.executable, "-m", "tests.integration.feddyn_count_worker",
        str(tmp_path / "run"), str(output), mode,
    ]
    completed = subprocess.run(
        ["zsh", "-lc", shlex.join(command)],
        capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text())
    assert result["actual_count"] == 2 and result["rejected_count"] == 3
    assert len(result["retries"]) == 3


@pytest.mark.parametrize("spawn", [False, True])
def test_parent_realizes_partition_once_without_using_backing_length(tmp_path, spawn):
    # The parent object records get() calls; the worker must receive this
    # same realized two-index partition, not draw a new partition itself.
    with configure_environment(configuration(spawn=spawn), runtime_root=tmp_path):
        s, c = server(), client(1)
        response, payload = dispatch(s, [1])[1]
        c.current_round = c._context.current_round = response["current_round"]
        c.lifecycle_strategy.process_server_response(c._context, response)
        c.training_strategy.load_payload(c._context, payload)
        context = c._context
        context.trainset = TensorDataset(
            torch.ones(100, 1).double(), torch.zeros(100, 1).double()
        )

        context.sampler = SingleDrawPartition()
        report, weights = asyncio.run(c.training_strategy.train(context))
        assert context.sampler.calls == 1
        assert report.num_samples == weights[1]["num_samples"] == 2
        assert c.trainer.model_update_strategy.result["num_samples"] == 2
        assert weights[0]["theta"].item() == pytest.approx(1.622, abs=1e-12)


@pytest.mark.parametrize("mode", ["uniform", "sample"])
@pytest.mark.parametrize("missing_client", [1, 2])
@pytest.mark.parametrize("previous_pending", [False, True])
def test_nonzero_history_requires_saved_observed_count(
    tmp_path, mode, missing_client, previous_pending
):
    with configure_environment(configuration(mode), runtime_root=tmp_path):
        s = server()
        for i in (1, 2):
            s.updates = [local(client(i), dispatch(s, [i])[i], (0.0, 4.0)[i - 1], 2)]
            asyncio.run(s._process_reports())
        s.save_to_checkpoint()
        path = Path(s.checkpoint_bundle_path())
        original_bytes = path.read_bytes()
        bundle = torch.load(path, weights_only=True)
        assert torch.count_nonzero(bundle["histories"][missing_client]["theta"])
        resumed = server()
        if previous_pending:
            resumed._resume_from_checkpoint()
        before = committed_state(resumed)
        authority = copy.deepcopy(
            (resumed.run_id, resumed.current_round, resumed.settings, resumed.schema)
        )
        pending = copy.deepcopy(resumed._pending_resume_rng)
        rng = resumed._rng_snapshot()
        snapshot = copy.deepcopy(resumed._committed_snapshot)
        del bundle["counts"][missing_client]
        torch.save(bundle, path)
        invalid_bytes = path.read_bytes()
        with pytest.raises(ValueError, match="FedDyn.*history.*count"):
            resumed._resume_from_checkpoint()
        assert_state_equal(committed_state(resumed), before)
        assert_state_equal(
            (resumed.run_id, resumed.current_round, resumed.settings, resumed.schema),
            authority,
        )
        assert_state_equal(resumed._pending_resume_rng, pending)
        assert_state_equal(resumed._committed_snapshot, snapshot)
        assert_state_equal(resumed._rng_snapshot(), rng)
        assert path.read_bytes() == invalid_bytes
        path.write_bytes(original_bytes)
        resumed._resume_from_checkpoint()
        assignment = dispatch(resumed, [missing_client])[missing_client]
        assert assignment[1][1]["expected_count_or_null"] == 2
        with pytest.raises(ValueError):
            local(client(missing_client), assignment, 0.0, 3)
        update = local(client(missing_client), assignment, 0.0, 2)
        resumed.updates = [update]
        asyncio.run(resumed._process_reports())
        assert resumed.committed_round == 3


@pytest.mark.parametrize("event", ["on_weights_received", "on_weights_aggregated"])
@pytest.mark.parametrize("mutation", ["bool-value", "float-value", "numpy-value", "float-key"])
def test_inactive_callback_count_types_rejected_before_commit(tmp_path, event, mutation):
    with configure_environment(configuration(), runtime_root=tmp_path):
        s = server()
        s.updates = [local(client(2), dispatch(s, [2])[2], 4.0, 1)]
        asyncio.run(s._process_reports())
        s.save_to_checkpoint()
        path = Path(s.checkpoint_bundle_path())
        canonical = path.read_bytes()
        update = local(client(1), dispatch(s, [1])[1], 0.0, 1)
        s.updates = [update]
        before = committed_state(s)
        snapshot = copy.deepcopy(s._committed_snapshot)

        def mutate(server, *args):
            assert type(server.observed_counts[2]) is int
            if mutation == "float-key":
                value = server.observed_counts.pop(2)
                server.observed_counts[2.0] = value
            else:
                server.observed_counts[2] = {
                    "bool-value": True, "float-value": 1.0,
                    "numpy-value": np.int64(1),
                }[mutation]

        callback = ServerCallback()
        setattr(callback, event, mutate)
        s.callback_handler.add_callback(callback)
        with pytest.raises(ValueError, match="FedDyn.*metadata"):
            asyncio.run(s._process_reports())
        assert_state_equal(committed_state(s), before)
        assert_state_equal(s._committed_snapshot, snapshot)
        assert type(s.observed_counts[2]) is int
        assert all(type(k) is int for k in s.observed_counts)
        assert s._pending is None and s._pending_seal is None
        assert path.read_bytes() == canonical
        s.callback_handler.callbacks.remove(callback)
        asyncio.run(s._process_reports())
        assert s.committed_round == 2 and s.observed_counts == {1: 1, 2: 1}
        s.save_to_checkpoint()
        assert torch.load(path, weights_only=True)["committed_round"] == 2
