import torch


def _ensure_tuple(sample):
    """Normalize sample to (feature, target) tuple, discarding extras."""
    if isinstance(sample, tuple):
        if len(sample) >= 2:
            return sample[0], sample[1]
        if len(sample) == 1:
            return sample[0], torch.zeros(1)
    return sample, torch.zeros(1)


class FeatureDataset(torch.utils.data.Dataset):
    """Used to prepare a feature dataset for a DataLoader in PyTorch."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        feature, target = _ensure_tuple(sample)
        return feature, target
