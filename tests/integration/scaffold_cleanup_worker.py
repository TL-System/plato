"""Actual spawned SCAFFOLD cleanup rejection and same-client retry."""

import json
import pickle
import sys
from pathlib import Path

from plato.trainers.strategies.algorithms.scaffold_strategy import (
    SCAFFOLDUpdateStrategy,
)
from tests.integration.test_scaffold_round_flow import (
    QuadraticTrainer,
    ScaffoldCallback,
    create_client,
    local_round,
    scaffold_client,
    shipped_config,
)
from tests.integration.utils import configure_environment
from tests.trainers.test_scaffold_strategy import ScalarModel, scalar_controls


class FailedCleanup(SCAFFOLDUpdateStrategy):
    fail = True

    def on_train_cleanup(self, context, successful):
        super().on_train_cleanup(context, successful)
        if successful and self.fail:
            raise RuntimeError("Deliberate successful-run cleanup failure")


class CleanupTrainer(QuadraticTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.model_update_strategy = FailedCleanup()
        self.model_update_strategy.setup(self.context)


def scenario(root, existing):
    with configure_environment(shipped_config(spawn=True), runtime_root=root):
        c = scaffold_client.create_client(
            model=ScalarModel, trainer=CleanupTrainer, callbacks=[ScaffoldCallback]
        )
        c.client_id = c._context.client_id = 1
        c.configure()
        strategy = c.trainer.model_update_strategy
        strategy.client_control_variate = scalar_controls(1)
        canonical = Path(strategy.client_control_variate_path)
        if existing:
            canonical.write_bytes(pickle.dumps(scalar_controls(1)))
        before = canonical.read_bytes() if existing else None
        try:
            local_round(
                c,
                [scalar_controls(1), scalar_controls(2)],
                round_id=1,
                target=0,
                samples=2,
            )
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("Spawned cleanup failure was accepted")
        assert strategy.client_control_variate["theta"].item() == 1
        assert (canonical.read_bytes() if canonical.exists() else None) == before
        try:
            strategy.get_update_payload(c.trainer.context)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Failed spawned result was outbound eligible")
        for i in (2, 1):
            c.client_id = c._context.client_id = i
            c.configure()
        fresh = create_client(1).trainer.model_update_strategy.client_control_variate
        if existing:
            assert fresh["theta"].item() == 1
        else:
            assert fresh is None
            strategy.client_control_variate = scalar_controls(1)
        strategy.fail = False
        _, payload = local_round(
            c, [scalar_controls(1), scalar_controls(2)], round_id=2, target=0, samples=2
        )
        assert abs(c.trainer.model.theta.item() - 0.62) < 1e-12
        assert abs(strategy.client_control_variate["theta"].item() - 0.9) < 1e-12
        assert abs(payload[1]["theta"].item() + 0.1) < 1e-12
        assert "complete_optimizer_step" not in c.trainer.context.state
        return dict(
            y=c.trainer.model.theta.item(),
            ci=strategy.client_control_variate["theta"].item(),
            delta=payload[1]["theta"].item(),
        )


if __name__ == "__main__":
    root, output = map(Path, sys.argv[1:3])
    root.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scenario(root, sys.argv[3] == "1"), indent=2))
