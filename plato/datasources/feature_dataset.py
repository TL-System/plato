import torch


class FeatureDataset(torch.utils.data.Dataset):
    """Used to prepare a feature dataset for a DataLoader in PyTorch."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]

        if isinstance(sample, (list, tuple)):
            if len(sample) >= 2:
                feature, label = sample[0], sample[1]
            elif len(sample) == 1:
                feature = sample[0]
                label = torch.tensor(0, dtype=torch.long)
            else:
                raise ValueError("Empty sample encountered in FeatureDataset.")
        else:
            feature = sample
            label = torch.tensor(0, dtype=torch.long)

        if torch.is_tensor(label) and label.dim() > 0:
            label = label.squeeze()

        return feature, label
