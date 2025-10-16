import torch


def _ensure_tuple(sample):
    """Normalize sample to (feature, target) tuple, discarding extras."""
    if isinstance(sample, tuple):
        if len(sample) >= 2:
            feature = sample[0]
            target = sample[1]
        elif len(sample) == 1:
            feature = sample[0]
            target = torch.zeros(1)
        else:
            feature = torch.zeros(1)
            target = torch.zeros(1)
    elif isinstance(sample, torch.Tensor):
        feature = sample
        target = torch.zeros(1)
    else:
        feature = torch.tensor(sample)
        target = torch.zeros(1)

    if torch.is_tensor(target) and target.ndim == 0:
        target = target.unsqueeze(0)

    return feature, target


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
