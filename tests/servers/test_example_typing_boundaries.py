"""Behavioral checks for maintained example input and scalar boundaries."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from plato.servers.strategies.aggregation.fedavg import FedAvgAggregationStrategy

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def knot_module(temp_config, monkeypatch):
    from plato.servers import fedavg

    # These checks exercise input conversion before any optional solver call.
    # The optimizer and membership inference dependencies have separate profiles.
    unlearning = ModuleType("fedunlearning_server")
    setattr(unlearning, "Server", fedavg.Server)
    cvxopt = ModuleType("cvxopt")

    def matrix(values, size=None):
        array = np.asarray(values)
        return array if size is None else array.reshape(size, order="F")

    setattr(cvxopt, "matrix", matrix)
    monkeypatch.setitem(sys.modules, "fedunlearning_server", unlearning)
    monkeypatch.setitem(sys.modules, "cvxopt", cvxopt)
    monkeypatch.setitem(sys.modules, "solver", ModuleType("solver"))
    return load_module(
        "knot_boundary_server", EXAMPLES / "unlearning/knot/knot_server.py"
    )


def knot_server(module):
    server = module.Server.__new__(module.Server)
    server.clusters = {1: None, 2: None}
    server.num_clusters = 2
    server.clients_similarity = {1: 0.25, 2: 0.75}
    return server


def test_knot_solver_assignment_uses_python_scalar_ids(knot_module):
    server = knot_server(knot_module)
    observed = []
    server.algorithm = SimpleNamespace(init_clusters=observed.append)
    server._convert_from_solver(np.array([[0, 1], [1, 0]], dtype=np.int64))
    assert server.clusters == {2: 0, 1: 1}
    assert all(
        type(key) is int and type(value) is int
        for key, value in server.clusters.items()
    )
    assert observed == [server.clusters]


@pytest.mark.parametrize("value", [None, "0.5", float("nan"), float("inf")])
def test_knot_rejects_incomplete_similarity_population(knot_module, value):
    server = knot_server(knot_module)
    server.clients_similarity[2] = value
    with pytest.raises(ValueError, match="similarity for every client"):
        server._convert_to_solver({1: 1.0, 2: 2.0})


@pytest.mark.parametrize("value", [None, "1", float("nan"), float("inf")])
def test_knot_rejects_invalid_training_times(knot_module, value):
    server = knot_server(knot_module)
    with pytest.raises(ValueError, match="training times"):
        server._convert_to_solver({1: 1.0, 2: value})


def test_knot_rejects_empty_population_and_zero_clusters(knot_module):
    server = knot_server(knot_module)
    with pytest.raises(ValueError, match="nonempty client population"):
        server._convert_to_solver({})
    server.num_clusters = 0
    with pytest.raises(ValueError, match="positive cluster count"):
        server._convert_to_solver({1: 1.0, 2: 2.0})


def test_knot_numeric_conversion_preserves_distance_matrix(knot_module):
    server = knot_server(knot_module)
    actual = server._convert_to_solver({1: np.float64(1), 2: np.float64(3)})
    expected = [np.hypot(0, 0), np.hypot(1, 0.25), np.hypot(2, 0.5), np.hypot(1, 0.25)]
    assert np.asarray(actual).ravel(order="F") == pytest.approx(expected)


@pytest.fixture
def feddf_modules(temp_config, monkeypatch):
    directory = EXAMPLES / "server_aggregation/feddf"
    monkeypatch.syspath_prepend(str(directory))
    # Load siblings with their real identities, retaining the production imports.
    modules = {}
    for name in (
        "feddf_utils",
        "feddf_algorithm",
        "feddf_server_strategy",
        "feddf_server",
    ):
        module = load_module(name, directory / (name + ".py"))
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
    return modules


class UnsizedProxy(Dataset):
    def __getitem__(self, index):
        return torch.ones(2)


def test_feddf_rejects_unsized_proxy_before_sampling_or_distillation(feddf_modules):
    dataset = UnsizedProxy()
    with pytest.raises(TypeError, match="sized proxy dataset"):
        feddf_modules["feddf_utils"].select_proxy_subset(dataset, size=1, seed=17)
    algorithm = feddf_modules["feddf_algorithm"].Algorithm.__new__(
        feddf_modules["feddf_algorithm"].Algorithm
    )
    with pytest.raises(TypeError, match="sized proxy dataset"):
        algorithm.distill_weights(
            {},
            torch.ones(1, 2),
            dataset,
            temperature=1,
            distillation_epochs=1,
            distillation_batch_size=1,
            distillation_learning_rate=0.1,
            distillation_optimizer_name="SGD",
            use_cosine_annealing=False,
            shuffle_batches=False,
        )


def test_feddf_rejects_incompatible_aggregation_strategy(feddf_modules):
    server = feddf_modules["feddf_server"].Server.__new__(
        feddf_modules["feddf_server"].Server
    )
    server.aggregation_strategy = FedAvgAggregationStrategy()
    with pytest.raises(TypeError, match="FedDFAggregationStrategy"):
        server.customize_server_payload({})
