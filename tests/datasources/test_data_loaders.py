"""Tests for the parallel and sequential data loader wrappers."""

import torch
from torch.utils.data import DataLoader, Dataset

from plato.utils import data_loaders


class ToyDataset(Dataset):
    """Dataset returning simple tensor-label pairs."""

    def __init__(self, length: int, input_dim: int = 6):
        torch.manual_seed(123)
        self.inputs = torch.randn(length, input_dim)
        self.labels = torch.randint(0, 3, (length,))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


def _build_loader(length: int, batch_size: int) -> DataLoader:
    dataset = ToyDataset(length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_parallel_data_loader_yields_batches_in_lockstep():
    """ParallelDataLoader should iterate in sync across wrapped loaders."""
    loader_small = _build_loader(length=16, batch_size=4)
    loader_large = _build_loader(length=16, batch_size=8)

    parallel_loader = data_loaders.ParallelDataLoader(
        [loader_small, loader_large, None]
    )

    assert len(parallel_loader) == min(len(loader_small), len(loader_large))

    for batch_id, (small_batch, large_batch) in enumerate(parallel_loader):
        if batch_id % 2 != 0:
            continue
        assert small_batch[0].shape[0] == 4
        assert large_batch[0].shape[0] == 8


def test_sequential_data_loader_iterates_through_all_batches():
    """SequentialDataLoader should exhaust loaders one after another."""
    loader_a = _build_loader(length=12, batch_size=3)
    loader_b = _build_loader(length=8, batch_size=2)

    sequence_loader = data_loaders.SequentialDataLoader([loader_a, loader_b, None])

    assert len(sequence_loader) == len(loader_a) + len(loader_b)

    seen = 0
    for batch in sequence_loader:
        seen += 1
        assert isinstance(batch, (list, tuple))
        assert batch[0].ndim == 2
    assert seen == len(sequence_loader)
