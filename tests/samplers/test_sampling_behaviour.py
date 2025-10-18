"""Behaviour-focused tests for non-IID sampler implementations."""

import pytest

from plato.config import Config
from plato.samplers import dirichlet, orthogonal


class _MockDataset:
    def __init__(self, targets):
        self.targets = targets
        self.classes = sorted(set(targets))

    def __len__(self):
        return len(self.targets)


class _ToyDatasource:
    def __init__(self, targets):
        self._train = _MockDataset(targets)
        self._test = _MockDataset(list(reversed(targets)))

    def targets(self):
        return self._train.targets

    def classes(self):
        return self._train.classes

    def get_train_set(self):
        return self._train

    def get_test_set(self):
        return self._test

    def num_train_examples(self):
        return len(self._train)


@pytest.fixture
def _reset_data_sections(temp_config):
    """Ensure Config.data and Config.algorithm can be updated safely."""
    original_data = temp_config.data._asdict()
    original_algorithm = temp_config.algorithm._asdict()
    yield temp_config
    Config.data = Config.namedtuple_from_dict(original_data)
    Config.algorithm = Config.namedtuple_from_dict(original_algorithm)


def test_dirichlet_sampler_is_deterministic_per_client(_reset_data_sections):
    """Dirichlet sampler should emit reproducible partitions per client."""
    data_dict = Config.data._asdict()
    data_dict.update({"partition_size": 6, "random_seed": 123, "concentration": 0.5})
    Config.data = Config.namedtuple_from_dict(data_dict)

    datasource = _ToyDatasource(targets=[0, 0, 1, 1, 2, 2, 0, 1, 2])

    sampler_one = dirichlet.Sampler(datasource, client_id=1, testing=False)
    indices_first = list(sampler_one.get())
    assert len(indices_first) == Config().data.partition_size
    assert set(indices_first).issubset(range(datasource.num_train_examples()))

    sampler_repeat = dirichlet.Sampler(datasource, client_id=1, testing=False)
    assert indices_first == list(sampler_repeat.get())

    sampler_two = dirichlet.Sampler(datasource, client_id=2, testing=False)
    assert indices_first != list(sampler_two.get())


def test_dirichlet_sampler_uses_test_targets_when_testing(_reset_data_sections):
    """Testing flag should swap to the test set target distribution."""
    data_dict = Config.data._asdict()
    data_dict.update({"partition_size": 4, "random_seed": 321, "concentration": 1.0})
    Config.data = Config.namedtuple_from_dict(data_dict)

    datasource = _ToyDatasource(targets=[0, 0, 1, 1, 2, 2, 0, 1, 2, 2])

    sampler = dirichlet.Sampler(datasource, client_id=1, testing=True)
    indices = list(sampler.get())

    # The test set reverses the targets, so the sampled indices should reflect that.
    test_targets = datasource.get_test_set().targets
    sampled_labels = [test_targets[index] for index in indices]
    assert len(indices) == Config().data.partition_size
    assert set(sampled_labels).issubset(set(test_targets))


def test_orthogonal_sampler_restricts_class_assignments(_reset_data_sections):
    """Orthogonal sampler should only draw data from the configured class slices."""
    data_dict = Config.data._asdict()
    data_dict.update(
        {
            "partition_size": 4,
            "random_seed": 99,
            "institution_class_ids": "0,1;2,3",
        }
    )
    Config.data = Config.namedtuple_from_dict(data_dict)

    algorithm_dict = Config.algorithm._asdict()
    algorithm_dict.update({"total_silos": 2})
    Config.algorithm = Config.namedtuple_from_dict(algorithm_dict)

    targets = [0, 0, 1, 1, 2, 2, 3, 3]
    datasource = _ToyDatasource(targets=targets)

    sampler_client1 = orthogonal.Sampler(datasource, client_id=1, testing=False)
    indices_c1 = list(sampler_client1.get())
    assert len(indices_c1) == Config().data.partition_size
    assert all(targets[idx] in {0, 1} for idx in indices_c1)

    sampler_client2 = orthogonal.Sampler(datasource, client_id=2, testing=False)
    indices_c2 = list(sampler_client2.get())
    assert len(indices_c2) == Config().data.partition_size
    assert all(targets[idx] in {2, 3} for idx in indices_c2)
