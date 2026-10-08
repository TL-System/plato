"""F17 must reject malformed native trees before arithmetic or mutation."""

import copy
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from plato.algorithms.mlx_fedavg import Algorithm
from plato.models.mlx.lenet5 import LeNet5
from plato.trainers.base import Trainer
from plato.trainers.mlx import ComposableMLXTrainer
from tests.mlx_native.helpers import assert_tree_equal, native_config


@pytest.mark.parametrize("operation", ["delta", "update", "load", "trainer_load"])
@pytest.mark.parametrize(
    "damage", ["missing", "extra", "broadcast", "shape", "none", "container"]
)
def test_lenet_rejects_tree_before_mutation(tmp_path, operation, damage):
    with native_config(tmp_path):
        trainer = ComposableMLXTrainer(model=LeNet5)
        algorithm = Algorithm(trainer)
        baseline = algorithm.extract_weights()
        malformed = copy.deepcopy(baseline)
        malformed["conv1"]["weight"] += 1
        if damage == "missing":
            del malformed["fc3"]["bias"]
        elif damage == "extra":
            malformed["fc3"]["extra"] = np.zeros(1, dtype=np.float32)
        elif damage == "broadcast":
            malformed["fc3"]["bias"] = np.zeros(1, dtype=np.float32)
        elif damage == "shape":
            malformed["fc3"]["bias"] = np.zeros(11, dtype=np.float32)
        elif damage == "none":
            malformed["fc3"]["bias"] = None
        else:
            malformed["fc3"] = list(malformed["fc3"].values())
        with pytest.raises(ValueError, match="fc3"):
            if operation == "delta":
                algorithm.compute_weight_deltas(baseline, [baseline, malformed])
            elif operation == "update":
                algorithm.update_weights(malformed)
            elif operation == "load":
                algorithm.load_weights(malformed)
            else:
                trainer._apply_model_state(malformed)
        assert_tree_equal(algorithm.extract_weights(), baseline)


@pytest.mark.parametrize(
    "other",
    [[], [np.zeros(2), np.zeros(2)], (np.zeros(2),), [None]],
    ids=["short", "long", "tuple-for-list", "none-for-array"],
)
def test_exact_sequence_and_none_contract(other):
    algorithm = Algorithm(cast(Trainer, SimpleNamespace(model=None)))
    with pytest.raises(ValueError):
        algorithm.compute_weight_deltas({"x": [np.zeros(2)]}, [{"x": other}])


def test_extracted_host_weights_own_snapshots(tmp_path):
    with native_config(tmp_path):
        trainer = ComposableMLXTrainer(model=LeNet5)
        algorithm = Algorithm(trainer)
        snapshot = algorithm.extract_weights()
        model = cast(LeNet5, trainer.model)
        live_view = np.asarray(model.conv1.weight)
        assert not np.shares_memory(snapshot["conv1"]["weight"], live_view)
        original = live_view.copy()
        snapshot["conv1"]["weight"] += 5
        np.testing.assert_array_equal(np.asarray(model.conv1.weight), original)


def test_array_cannot_replace_none_leaf():
    algorithm = Algorithm(cast(Trainer, SimpleNamespace(model=None)))
    with pytest.raises(ValueError, match="optional.*None"):
        algorithm.compute_weight_deltas({"optional": None}, [{"optional": np.zeros(2)}])


def test_matching_nested_none_and_sequences():
    algorithm = Algorithm(cast(Trainer, SimpleNamespace(model=None)))
    baseline = {
        "x": [np.array([2], dtype=np.float32), None],
        "y": (np.array([3], dtype=np.float32),),
    }
    current = {
        "x": [np.array([4], dtype=np.float32), None],
        "y": (np.array([7], dtype=np.float32),),
    }
    delta = algorithm.compute_weight_deltas(baseline, [current])[0]
    assert_tree_equal(
        delta,
        {
            "x": [np.array([2], dtype=np.float32), None],
            "y": (np.array([4], dtype=np.float32),),
        },
    )


def test_all_received_trees_preflight_before_first_subtraction(monkeypatch):
    from plato.algorithms import mlx_fedavg

    algorithm = Algorithm(cast(Trainer, SimpleNamespace(model=None)))
    baseline = {"early": np.zeros(2), "late": np.zeros(2)}
    received = [copy.deepcopy(baseline), {"early": np.zeros(2), "late": np.zeros(1)}]
    conversions = []

    def observe(value):
        conversions.append(value)
        return value

    monkeypatch.setattr(mlx_fedavg, "_to_numpy", observe)
    with pytest.raises(ValueError, match="late.*shape"):
        algorithm.compute_weight_deltas(baseline, received)
    assert conversions == []
