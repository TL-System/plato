"""Initialization, logical-client replay and caller RNG ownership regressions."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch

from plato.algorithms.mlx_fedavg import Algorithm
from plato.config import Config
from plato.models.mlx.lenet5 import LeNet5
from plato.trainers.mlx import ComposableMLXTrainer
from tests.mlx_native.helpers import assert_tree_equal, dataset, native_config


def run_replay(path, identities, seed=29):
    command = [
        sys.executable,
        "-m",
        "tests.mlx_native.seed_probe",
        str(path),
        json.dumps(identities),
        str(seed),
    ]
    result = subprocess.run(
        command,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(path.read_text())


def test_fresh_and_reassigned_workers_replay_same_logical_streams(tmp_path):
    identities = [[1, 1], [2, 1], [1, 2]]
    reused = run_replay(tmp_path / "reused.json", identities)
    reordered = run_replay(tmp_path / "reordered.json", list(reversed(identities)))
    assert reused == list(reversed(reordered))
    fresh = [
        run_replay(tmp_path / f"fresh-{i}.json", [identity])[0]
        for i, identity in enumerate(identities)
    ]
    assert fresh == reused
    for result in reused:
        assert sorted(result["order"]) == list(range(9))
        assert len(result["losses"]) == 3
        assert result["weights"] != result["initial"]
    assert reused[0]["masks"] != reused[1]["masks"]
    assert reused[0]["masks"] != reused[2]["masks"]
    assert reused[0]["transforms"] != reused[1]["transforms"]
    changed_seed = run_replay(tmp_path / "other-seed.json", [[1, 1]], seed=31)[0]
    assert changed_seed["initial"] == reused[0]["initial"]
    assert changed_seed["weights"] != reused[0]["weights"]


@pytest.mark.parametrize("failure", [False, True], ids=["success", "dataset-failure"])
def test_seeded_factory_and_training_restore_all_caller_rngs(tmp_path, failure):
    random.seed(81)
    np.random.seed(82)
    torch.manual_seed(83)
    mx.random.seed(84)
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    before_mlx = np.array(mx.random.state[0], copy=True)
    with native_config(tmp_path, model_seed=17, training_seed=29):
        trainer = ComposableMLXTrainer(model=LeNet5)
        samples = dataset()

        class FailingData:
            def __len__(self):
                return 8

            def __getitem__(self, index):
                random.random()
                np.random.uniform()
                torch.rand(1)
                mx.random.normal((1,))
                raise RuntimeError("dataset failure")

        if failure:
            with pytest.raises(RuntimeError, match="dataset failure"):
                trainer.train_model(Config().trainer._asdict(), FailingData(), None)
        else:
            trainer.train_model(Config().trainer._asdict(), samples, None)
        assert random.getstate() == before_python
        after_numpy = np.random.get_state()
        assert after_numpy[0] == before_numpy[0]
        np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
        assert after_numpy[2:] == before_numpy[2:]
        assert torch.equal(torch.get_rng_state(), before_torch)
        np.testing.assert_array_equal(np.asarray(mx.random.state[0]), before_mlx)


def test_initialization_seed_replays_and_unseeded_initialization_remains_legacy(
    tmp_path,
):
    with native_config(tmp_path / "seeded", model_seed=17):
        first = Algorithm(ComposableMLXTrainer(model=LeNet5)).extract_weights()
        second = Algorithm(ComposableMLXTrainer(model=LeNet5)).extract_weights()
        assert_tree_equal(first, second)
    with native_config(tmp_path / "unseeded"):
        before = np.array(mx.random.state[0], copy=True)
        first = Algorithm(ComposableMLXTrainer(model=LeNet5)).extract_weights()
        second = Algorithm(ComposableMLXTrainer(model=LeNet5)).extract_weights()
        assert not np.array_equal(first["conv1"]["weight"], second["conv1"]["weight"])
        assert not np.array_equal(np.asarray(mx.random.state[0]), before)
