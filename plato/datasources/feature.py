"""
The feature dataset server received from clients.
"""

from itertools import chain
import torch
from plato.datasources import base


class FeatureDataset(torch.utils.data.Dataset):
    """Used to prepare a feature dataset for a DataLoader in PyTorch."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label


class DataSource(base.DataSource):
    """The feature dataset."""
    def __init__(self, features):
        super().__init__()

        # Faster way to deep flatten a list of lists compared to list comprehension
        feature_dataset = list(chain.from_iterable(features))
        self.trainset = feature_dataset
        self.testset = []

    def torch_dataset(self):
        """ Prepare the feature dataset when PyTorch is used. """
        return FeatureDataset(self.trainset)
