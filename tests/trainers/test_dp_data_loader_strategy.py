import torch
from torch.utils.data import SubsetRandomSampler, TensorDataset

from plato.trainers.diff_privacy import DPDataLoaderStrategy
from plato.trainers.strategies.base import TrainingContext


class _FakePlatoSampler:
    """Minimal stub to mimic Plato sampler behaviour with subset indices."""

    def __init__(self, indices):
        self.subset_indices = indices

    def get(self):
        return SubsetRandomSampler(self.subset_indices)


def _collect_dataset_indices(loader):
    """Utility to gather indices from batches for assertions."""
    collected = []
    for values, _ in loader:
        collected.extend(values.tolist())
    return sorted(collected)


def test_dp_strategy_handles_plato_sampler_get():
    """DP data loader should resolve Plato sampler objects into subset indices."""
    dataset = TensorDataset(torch.arange(10), torch.arange(10))
    sampler = _FakePlatoSampler([1, 3, 5, 7])
    context = TrainingContext()

    loader = DPDataLoaderStrategy().create_train_loader(
        dataset, sampler, batch_size=2, context=context
    )

    assert _collect_dataset_indices(loader) == [1, 3, 5, 7]


def test_dp_strategy_handles_torch_sampler_directly():
    """DP data loader should accept native PyTorch samplers."""
    dataset = TensorDataset(torch.arange(8), torch.arange(8))
    torch_sampler = SubsetRandomSampler([0, 2, 4, 6])
    context = TrainingContext()

    loader = DPDataLoaderStrategy().create_train_loader(
        dataset, torch_sampler, batch_size=2, context=context
    )

    assert _collect_dataset_indices(loader) == [0, 2, 4, 6]
