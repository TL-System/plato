"""Guard the real single-partition parent/worker oracle from pytest imports."""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

from plato.callbacks.trainer import TrainerCallback
from tests.integration.test_feddyn_count_boundaries import _run_single_partition


class ObservePartition(TrainerCallback):
    """Record the actual sampler from each epoch of the spawned worker."""

    def __init__(self, path):
        self.path = path
        self.epochs = []

    def on_train_epoch_start(self, trainer, config, **kwargs):
        sampler = trainer.train_loader.sampler
        assert list(sampler.indices) == [7, 13]
        assert len(sampler) == 2
        assert len(trainer.trainset) == 100
        self.epochs.append(
            {
                "pid": os.getpid(),
                "start_method": mp.get_start_method(),
                "indices": list(sampler.indices),
                "backing_items": len(trainer.trainset),
                "count": trainer.context.state["feddyn_attempt_count"],
            }
        )
        self.path.write_text(json.dumps(self.epochs, indent=2))


def run(root):
    observation = root / "worker-partition.json"
    result = _run_single_partition(
        root, spawn=True, observer=ObservePartition(observation)
    )
    epochs = json.loads(observation.read_text())
    assert len(epochs) == 2
    for epoch in epochs:
        assert epoch["pid"] != os.getpid()
        assert epoch["start_method"] == "spawn"
        assert epoch["indices"] == [7, 13]
        assert epoch["backing_items"] == 100 and epoch["count"] == 2
    assert not mp.active_children()
    return {**result, "worker_epochs": len(epochs), "no_active_children": True}


if __name__ == "__main__":
    from tests.integration.feddyn_partition_worker import run as guarded_run

    root, output = map(Path, sys.argv[1:3])
    root.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(guarded_run(root), indent=2))
