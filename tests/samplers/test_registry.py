"""Tests for sampler registry implementations."""

import torch
from torch.utils.data import Dataset

from plato.samplers import registry as samplers_registry


class ToyDataset(Dataset):
    """Dataset exposing deterministic indices and labels."""

    def __init__(self, length: int = 8):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.tensor([index], dtype=torch.float32), index % 2


class ToyDatasource:
    """Datasource compatible with the sampler registry helpers."""

    def __init__(self):
        self._train = ToyDataset()
        self._test = ToyDataset(length=4)

    def num_train_examples(self):
        return len(self._train)

    def get_train_set(self):
        return self._train

    def get_test_set(self):
        return self._test


def _materialise_indices(sampler):
    """Helper to convert a torch sampler to a list of indices."""
    return list(iter(sampler))


def test_iid_sampler_is_deterministic_per_client(temp_config):
    """Repeated calls for the same client should produce identical partitions."""
    datasource = ToyDatasource()
    sampler = samplers_registry.get(datasource, client_id=1)
    indices_first = _materialise_indices(sampler.get())
    sampler = samplers_registry.get(datasource, client_id=1)
    indices_second = _materialise_indices(sampler.get())

    assert indices_first == indices_second


def test_iid_sampler_assigns_distinct_partitions(temp_config):
    """Different clients should receive different (or permuted) subsets."""
    datasource = ToyDatasource()
    sampler_one = samplers_registry.get(datasource, client_id=1)
    sampler_two = samplers_registry.get(datasource, client_id=2)

    indices_one = set(_materialise_indices(sampler_one.get()))
    indices_two = set(_materialise_indices(sampler_two.get()))

    assert indices_one != indices_two


def test_sampler_reports_partition_size(temp_config):
    """Sampler metadata should reflect the configured partition size."""
    datasource = ToyDatasource()
    sampler = samplers_registry.get(datasource, client_id=1)

    assert sampler.num_samples() == 4
