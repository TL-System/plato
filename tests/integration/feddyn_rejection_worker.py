"""Actual child save/export defects followed by parent rollback and retries."""

import asyncio
import copy
import json
import multiprocessing as mp
import pickle
import sys
from fractions import Fraction
from pathlib import Path

import torch

from plato.config import Config
from plato.trainers.strategies.algorithms.feddyn_strategy import FedDynUpdateStrategy
from plato.utils.checkpoint_paths import checkpoint_name, checkpoint_path
from tests.integration.test_feddyn_round_flow import (
    QuadraticTrainer,
    assert_state_equal,
    client,
    configuration,
    dispatch,
    local,
    rational_round,
    server,
)
from tests.integration.utils import configure_environment


class FailedParentHandoff(FedDynUpdateStrategy):
    fail_parent = False

    def load_worker_state(self, state, context):
        super().load_worker_state(state, context)
        if self.fail_parent and mp.current_process().name == "MainProcess":
            raise RuntimeError("Deliberate parent handoff failure")


class BrokenWorker(QuadraticTrainer):
    defect = None

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.model_update_strategy = FailedParentHandoff()
        self.model_update_strategy.setup(self.context)

    def save_model(self, *args, **kwargs):
        super().save_model(*args, **kwargs)
        if self.defect == "save":
            raise RuntimeError("Deliberate model-save failure")

    def train_process(self, config, trainset, sampler, **kwargs):
        super().train_process(config, trainset, sampler, **kwargs)
        defect = self.defect
        if defect is None or defect == "parent":
            return
        path = Path(self._training_state_path(config["run_id"]))
        if defect in ("exit", "save"):
            raise RuntimeError("Deliberate post-save worker failure")
        if defect == "missing":
            path.unlink()
            return
        if defect == "corrupt":
            path.write_bytes(b"not a worker state")
            return
        if defect == "model":
            name = checkpoint_name(
                Config().trainer.model_name,
                self.client_id,
                config["run_id"],
                suffix=".safetensors",
            )
            Path(checkpoint_path(Config.params["model_path"], name)).write_bytes(
                b"not a model"
            )
            return
        envelope = pickle.loads(path.read_bytes())
        state = envelope["state"]
        if defect == "token":
            envelope["token"] = "stale"
        elif defect == "client":
            envelope["client_id"] = 2
        elif defect == "round":
            state["dispatch"]["metadata"]["round"] += 1
        elif defect == "key":
            state.pop("history")
        elif defect == "shape":
            state["history"]["theta"] = torch.tensor(0.0, dtype=torch.double)
        elif defect == "nan":
            state["history"]["theta"].fill_(float("nan"))
        elif defect == "equation":
            state["history"]["theta"].add_(1)
        else:
            raise AssertionError(defect)
        path.write_bytes(pickle.dumps(envelope))


def scenario(root, defect):
    with configure_environment(configuration(spawn=True), runtime_root=root):
        s, c = server(), client(1, BrokenWorker)
        s.save_to_checkpoint()
        canonical = Path(s.checkpoint_bundle_path()).read_bytes()
        assignment = dispatch(s, [1])[1]
        strategy = c.trainer.model_update_strategy
        # Bind the actual downlink once so failed training can be compared to
        # the same immutable dispatch, rather than a construction-time state.
        c.current_round = c._context.current_round = 1
        c.lifecycle_strategy.process_server_response(c._context, assignment[0])
        c.training_strategy.load_payload(c._context, assignment[1])
        baseline = copy.deepcopy(c.trainer.model.state_dict())
        original_dispatch = copy.deepcopy(strategy.dispatch)
        c.trainer.defect = defect
        if defect == "parent":
            strategy.fail_parent = True
        try:
            local(c, assignment, 0.0, 2)
        except (Exception, BaseExceptionGroup):
            pass
        else:
            raise AssertionError("Malformed child was accepted")
        assert_state_equal(c.trainer.model.state_dict(), baseline)
        assert_state_equal(strategy.dispatch, original_dispatch)
        assert strategy.result is None and not strategy.accepted
        assert c.trainer.model.theta.grad is None
        assert Path(s.checkpoint_bundle_path()).read_bytes() == canonical
        assert s.committed_round == 0 and all(
            h["theta"].item() == 0 for h in s.histories.values()
        )
        try:
            strategy.get_update_payload(c.trainer.context)
        except ValueError:
            pass
        else:
            raise AssertionError("Failed child was outbound eligible")
        strategy.fail_parent = False
        c.trainer.defect = None
        x, h = Fraction(2), [Fraction(0), Fraction(0)]
        records = []
        for i in (1, 2, 1):
            a = assignment if i == 1 and s.committed_round == 0 else dispatch(s, [i])[i]
            update = local(c, a, (0.0, 4.0)[i - 1], 2)
            x, h, endpoints = rational_round(
                x, h, [i], [Fraction(0), Fraction(4)], [2, 2], "uniform"
            )
            assert abs(update.payload[0]["theta"].item() - float(endpoints[i])) < 1e-12
            s.updates = [update]
            asyncio.run(s._process_reports())
            assert abs(s.trainer.model.theta.item() - float(x)) < 1e-12
            for j, expected in enumerate(h, 1):
                assert abs(s.histories[j]["theta"].item() - float(expected)) < 1e-12
            records.append(float(x))
        return records


if __name__ == "__main__":
    from tests.integration.feddyn_rejection_worker import scenario as guarded_scenario

    root, output = map(Path, sys.argv[1:3])
    root.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(guarded_scenario(root, sys.argv[3])))
