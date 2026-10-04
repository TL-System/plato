"""Partition, count, RNG and bounded failure regressions on real samplers."""

import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from plato.config import Config
from plato.samplers import dirichlet, iid, orthogonal, registry, sampler_utils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, targets):
        self.targets = targets
        self.classes = list(range(4))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return index, self.targets[index]


class Datasource:
    def __init__(self, targets=None):
        self.trainset = Dataset(list(range(4)) * 20 if targets is None else targets)
        self.testset = Dataset(list(reversed(self.trainset.targets)))

    def get_train_set(self):
        return self.trainset

    def get_test_set(self):
        return self.testset

    def classes(self):
        return self.trainset.classes

    def targets(self):
        return self.trainset.targets

    def get_modality_name(self):
        return ["rgb", "audio", "flow"]


def assert_numpy_state_equal(before, rng=None):
    after = (np.random if rng is None else rng).get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_dirichlet_realized_count_is_stable_and_matches_loader(
    temp_config, distribution
):
    Config.data.partition_size = 16
    Config.data.random_seed = 31
    Config.data.partition_distribution = Config.node_from_dict(
        {
            "distribution": distribution,
            "low": 0.4,
            "high": 0.2 if distribution == "normal" else 1.2,
            "mean": 1.0,
        }
    )
    source = Datasource()
    sampler = dirichlet.Sampler(source, client_id=2, testing=False)
    before = np.random.get_state()
    private_rng = getattr(sampler, "rng", np.random)
    private_before = private_rng.get_state()
    weights_before = list(sampler.sample_weights)
    torch_before = torch.get_rng_state()
    count = sampler.num_samples()
    assert [sampler.num_samples() for _ in range(10)] == [count] * 10
    assert_numpy_state_equal(before)
    assert_numpy_state_equal(private_before, private_rng)
    assert sampler.sample_weights == weights_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    indices = list(sampler.get())
    loaded = [
        int(i)
        for batch, _ in DataLoader(source.trainset, sampler=sampler.get(), batch_size=3)
        for i in batch
    ]
    assert loaded == indices
    assert len(indices) == count == len(set(indices))
    np.random.random(50)
    repeated = dirichlet.Sampler(source, client_id=2, testing=False)
    assert repeated.num_samples() == count
    assert list(repeated.get()) == indices


def test_empty_iid_fails_promptly_in_subprocess(temp_config):
    script = """
from plato.config import Config
from plato.samplers.iid import Sampler
Config._instance = object.__new__(Config)
Config.data = Config.node_from_dict({'partition_size': 4, 'random_seed': 1})
Config.clients = Config.node_from_dict({'total_clients': 2})
class Source:
    def get_train_set(self): return []
try:
    Sampler(Source(), 1, False)
except ValueError as exc:
    assert 'empty' in str(exc).lower(), str(exc)
    print('empty dataset rejected')
else:
    raise AssertionError('empty dataset silently accepted')
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
    except subprocess.TimeoutExpired:
        pytest.fail("IID sampling hung on empty data (5 second deadline)")
    assert result.returncode == 0, result.stderr
    assert "empty dataset rejected" in result.stdout


def test_iid_padding_and_membership_reference(temp_config):
    Config.data.partition_size = 4
    Config.data.random_seed = 1
    source = Datasource([0, 1, 2, 3, 0])
    # Legacy seed-1 shuffle is [2, 1, 4, 0, 3]; pad by repeating from the start,
    # then stride by two clients. This fixed reference protects valid behavior.
    first = iid.Sampler(source, 1, False)
    second = iid.Sampler(source, 2, False)
    assert first.subset_indices == [2, 4, 3, 1]
    assert second.subset_indices == [1, 0, 2, 4]
    assert first.num_samples() == second.num_samples() == 4


@pytest.mark.parametrize("name", list(registry.registered_samplers))
def test_seeded_sampler_construction_preserves_caller_rngs(temp_config, name):
    Config.data.update(
        {
            "random_seed": 12,
            "partition_size": 4,
            "min_partition_size": 4,
            "per_client_classes_size": 2,
            "anchor_classes": [0, 1],
            "keep_anchor_classes_size": 1,
            "consistent_clients": [0],
            "non_iid_clients": 1,
            "testset_size": 5,
        }
    )
    Config.algorithm.total_silos = 2
    before = np.random.get_state()
    python_before = random.getstate()
    torch_before = torch.get_rng_state()
    sampler = registry.get(Datasource(), 1, testing=True, sampler_type=name)
    assert_numpy_state_equal(before)
    assert random.getstate() == python_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    first = list(sampler.get())
    np.random.random(20)
    random.random()
    repeated = registry.get(Datasource(), 1, testing=True, sampler_type=name)
    assert list(repeated.get()) == first


@pytest.mark.parametrize(
    "name", ["label_quantity_noniid", "mixed_label_quantity_noniid"]
)
def test_full_class_partitions_cover_data_once_for_all_client_ids(temp_config, name):
    Config.data.per_client_classes_size = 4
    Config.data.anchor_classes = [0, 1, 2, 3]
    Config.data.keep_anchor_classes_size = 1
    Config.data.consistent_clients = [0]
    source = Datasource()
    partitions = [
        list(registry.get(source, client_id, sampler_type=name).get())
        for client_id in (1, 2)
    ]
    assert all(len(partition) == 40 for partition in partitions)
    assert sorted(partitions[0] + partitions[1]) == list(range(80))
    assert all(
        set(source.targets()[i] for i in part) == {0, 1, 2, 3} for part in partitions
    )


def test_label_quantity_without_anchors(temp_config):
    Config.data.per_client_classes_size = 2
    source = Datasource()
    partitions = [
        list(
            registry.get(source, client_id, sampler_type="label_quantity_noniid").get()
        )
        for client_id in (1, 2)
    ]
    assert all(len(set(source.targets()[i] for i in part)) == 2 for part in partitions)
    assert not set(partitions[0]).intersection(partitions[1])


def test_sample_quantity_condition_reports_selected_labels(temp_config):
    Config.data.min_partition_size = 4
    source = Datasource()
    sampler = registry.get(source, 1, sampler_type="sample_quantity_noniid")
    selected = [source.targets()[i] for i in sampler.get()]
    assert dict(sampler.get_sampled_data_condition()) == {
        label: selected.count(label) for label in set(selected)
    }


def test_empty_indices_cannot_be_extended():
    # The IID hang has a second active call site in quantity-skew padding.
    script = """
from plato.samplers.sampler_utils import extend_indices
try:
    extend_indices([], 1)
except ValueError as exc:
    assert 'empty' in str(exc).lower()
else:
    raise AssertionError('expected empty input rejection')
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script], timeout=5, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Quantity-skew padding hung on empty indices")
    assert result.returncode == 0, result.stderr


def test_orthogonal_rejects_size_exceeding_assigned_classes(temp_config):
    Config.algorithm.total_silos = 2
    Config.data.institution_class_ids = "0,1;2,3"
    Config.data.partition_size = 5
    # Only four of eight samples belong to this client's assigned classes.
    with pytest.raises(ValueError, match="assigned|eligible"):
        sampler = orthogonal.Sampler(Datasource(list(range(4)) * 2), 1, False)
        list(sampler.get())


@pytest.mark.parametrize(
    "name",
    [
        "iid",
        "noniid",
        "mixed",
        "distribution_noniid",
        "label_quantity_noniid",
        "mixed_label_quantity_noniid",
        "sample_quantity_noniid",
    ],
)
@pytest.mark.parametrize("client_id", [0, 3])
def test_partition_samplers_reject_out_of_range_ids(temp_config, name, client_id):
    Config.data.update(
        {
            "partition_size": 4,
            "min_partition_size": 4,
            "per_client_classes_size": 2,
            "anchor_classes": [0, 1],
            "keep_anchor_classes_size": 1,
            "consistent_clients": [0],
            "non_iid_clients": 1,
        }
    )
    with pytest.raises(ValueError, match="client_id"):
        registry.get(Datasource(), client_id, sampler_type=name)


def test_dirichlet_skew_zero_minimum_is_valid():
    proportions = sampler_utils.create_dirichlet_skew(
        20, 1.0, 2, min_partition_size=0, rng=np.random.RandomState(5)
    )
    assert len(proportions) == 2
    assert np.isfinite(proportions).all()
    assert sum(proportions) == pytest.approx(1.0)


def test_single_dirichlet_partition_can_use_entire_dataset():
    proportions = sampler_utils.create_dirichlet_skew(
        20, 1.0, 1, min_partition_size=20, rng=np.random.RandomState(5)
    )
    np.testing.assert_array_equal(proportions, [1.0])


def test_modality_quantity_single_modality_fallback(temp_config):
    class ImageOnlySource:
        pass

    sampler = registry.get(
        ImageOnlySource(), 1, sampler_type="modality_quantity_noniid"
    )
    assert list(sampler.get()) == ["rgb"]
    assert sampler.modality_size() == 1


def test_infeasible_dirichlet_minimum_fails_promptly():
    script = """
from plato.samplers.sampler_utils import create_dirichlet_skew
try:
    create_dirichlet_skew(10, 1.0, 2, min_partition_size=5)
except ValueError as exc:
    assert 'Minimum' in str(exc)
else:
    raise AssertionError('expected infeasible minimum rejection')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], timeout=5, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_mixed_label_partitions_respect_excluded_anchor_pool(temp_config):
    Config.data.update(
        {
            "per_client_classes_size": 2,
            "anchor_classes": [0, 1],
            "keep_anchor_classes_size": 0,
            "consistent_clients": [0],
        }
    )
    source = Datasource()
    first = registry.get(source, 1, sampler_type="mixed_label_quantity_noniid")
    second = registry.get(source, 2, sampler_type="mixed_label_quantity_noniid")
    assert set(source.targets()[i] for i in first.get()) == {0, 1}
    assert set(source.targets()[i] for i in second.get()) == {2, 3}
    assert sorted(list(first.get()) + list(second.get())) == list(range(80))


def test_infeasible_label_pool_fails_promptly():
    script = """
from plato.samplers.sampler_utils import assign_sub_classes
try:
    assign_sub_classes([0,1,2,3], [0,1,2,3], 2, 3, anchor_classes=[0,1,2],
                       consistent_clients=[0], keep_anchor_classes_size=0)
except ValueError as exc:
    assert 'pool' in str(exc)
else:
    raise AssertionError('expected infeasible class pool rejection')
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script], timeout=5, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Infeasible label pool caused an unbounded partition loop")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("edge_id", [4, 5])
def test_real_cross_silo_configure_accepts_assigned_edge_evaluation_id(
    temp_config, tmp_path, monkeypatch, edge_id
):
    monkeypatch.delenv("config_file", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "edge-test", "-c", "configs/MNIST/fedavg_cross_silo_lenet5.toml",
            "--base", str(tmp_path), "-i", str(edge_id), "-p", "8101",
        ],
    )
    Config.reset()
    Config()
    assert Config.is_edge_server()
    assert Config.clients.total_clients < Config.args.id <= (
        Config.clients.total_clients + Config.algorithm.total_silos
    )
    from plato.servers import fedavg, fedavg_cs

    Config.server.edge_do_test = True
    Config.data.testset_sampler = "noniid"
    Config.data.random_seed = 12
    source = Datasource([0, 1] * 500)
    server = object.__new__(fedavg_cs.Server)

    class Trainer:
        def set_client_id(self, client_id):
            self.client_id = client_id

    trainer = Trainer()
    monkeypatch.setattr(fedavg.Server, "configure", lambda self: None)
    monkeypatch.setattr(server, "init_trainer", lambda: None)
    monkeypatch.setattr(server, "require_trainer", lambda: trainer)
    monkeypatch.setattr(fedavg_cs.datasources_registry, "get", lambda **kw: source)
    monkeypatch.setattr(
        fedavg_cs.processor_registry, "get", lambda *a, **kw: (None, None)
    )
    before = np.random.get_state()
    server.configure()
    assert_numpy_state_equal(before)
    assert trainer.client_id == edge_id
    sampler = server.testset_sampler
    assert sampler is not None
    assert sampler.num_samples() == 600
    assert sampler is not None
    indices = list(sampler.get())
    assert len(indices) == len(set(indices)) == 600
    assert all(0 <= index < 1000 for index in indices)
    repeated = registry.get(source, edge_id, testing=True, sampler_type="noniid")
    assert list(repeated.get()) == indices


@pytest.mark.parametrize(
    "cross_silo,port,role_id,sampler_id,testing",
    [
        (False, 8101, 3, 3, True),
        (True, None, 3, 3, True),
        (True, 8101, 1, 3, True),
        (True, 8101, 3, 4, True),
        (True, 8101, 5, 5, True),
        (True, 8101, 3, 3, False),
    ],
)
def test_dirichlet_rejects_unassigned_edge_or_training_ids(
    temp_config, cross_silo, port, role_id, sampler_id, testing
):
    Config.algorithm.cross_silo = cross_silo
    Config.algorithm.total_silos = 2
    Config.args.port = port
    Config.args.id = role_id
    with pytest.raises(ValueError, match="client_id"):
        dirichlet.Sampler(Datasource(), sampler_id, testing)
