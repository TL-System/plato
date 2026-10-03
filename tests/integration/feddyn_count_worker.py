"""Real worker count corruption, parent rejection, and numerical retries."""

import asyncio
import copy
import json
import multiprocessing as mp
import pickle
import sys
from fractions import Fraction
from pathlib import Path

from tests.integration.test_feddyn_round_flow import (
    QuadraticTrainer,
    assert_state_equal,
    client,
    committed_state,
    configuration,
    dispatch,
    local,
    rational_round,
    server,
)
from tests.integration.utils import configure_environment


class CorruptCountTrainer(QuadraticTrainer):
    corrupt_count = True

    def train_process(self, config, trainset, sampler, **kwargs):
        super().train_process(config, trainset, sampler, **kwargs)
        if not self.corrupt_count:
            return
        path = Path(self._training_state_path(config["run_id"]))
        envelope = pickle.loads(path.read_bytes())
        assert envelope["state"]["num_samples"] == len(sampler) == 2
        envelope["state"]["num_samples"] = 3
        path.write_bytes(pickle.dumps(envelope))


def scenario(root, mode):
    with configure_environment(configuration(mode, (2, 2), True), runtime_root=root):
        s, c = server(), client(1, CorruptCountTrainer)
        assignment = dispatch(s, [1])[1]
        assert (assignment[1][1]["expected_count_or_null"] is None) == (
            mode == "uniform"
        )
        c.current_round = c._context.current_round = 1
        c.lifecycle_strategy.process_server_response(c._context, assignment[0])
        c.training_strategy.load_payload(c._context, assignment[1])
        baseline = copy.deepcopy(c.trainer.model.state_dict())
        parameter_ids = [id(p) for p in c.trainer.model.parameters()]
        prior_dispatch = copy.deepcopy(c.trainer.model_update_strategy.dispatch)
        before = committed_state(s)
        s.save_to_checkpoint()
        canonical = Path(s.checkpoint_bundle_path()).read_bytes()
        try:
            local(c, assignment, 0.0, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Parent accepted corrupted current-partition count")
        strategy = c.trainer.model_update_strategy
        assert not strategy.accepted and strategy.result is None
        assert_state_equal(c.trainer.model.state_dict(), baseline)
        assert_state_equal(strategy.dispatch, prior_dispatch)
        assert_state_equal(committed_state(s), before)
        assert [id(p) for p in c.trainer.model.parameters()] == parameter_ids
        assert c.trainer.model.theta.grad is None
        assert Path(s.checkpoint_bundle_path()).read_bytes() == canonical
        try:
            strategy.get_update_payload(c.trainer.context)
        except ValueError:
            pass
        else:
            raise AssertionError("Rejected result was outbound eligible")
        c.trainer.corrupt_count = False
        x, h, records = Fraction(2), [Fraction(0), Fraction(0)], []
        for i in (1, 2, 1):
            a = assignment if s.committed_round == 0 else dispatch(s, [i])[i]
            update = local(c, a, (0.0, 4.0)[i - 1], 2)
            x, h, endpoints = rational_round(
                x, h, [i], [Fraction(0), Fraction(4)], [2, 2], mode
            )
            assert abs(update.payload[0]["theta"].item() - float(endpoints[i])) < 1e-12
            assert update.report.num_samples == update.payload[1]["num_samples"] == 2
            s.updates = [update]
            asyncio.run(s._process_reports())
            assert abs(s.trainer.model.theta.item() - float(x)) < 1e-12
            for j, expected in enumerate(h, 1):
                assert abs(s.histories[j]["theta"].item() - float(expected)) < 1e-12
            records.append({"round": s.committed_round, "cloud": float(x)})
        assert not mp.active_children()
        return {"rejected_count": 3, "actual_count": 2, "retries": records}


if __name__ == "__main__":
    from tests.integration.feddyn_count_worker import scenario as guarded_scenario

    root, output = map(Path, sys.argv[1:3])
    root.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(guarded_scenario(root, sys.argv[3]), indent=2))
