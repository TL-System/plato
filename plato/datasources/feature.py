"""
The feature dataset server received from clients.
"""

from itertools import chain
from plato.datasources import base


class DataSource(base.DataSource):
    """The feature dataset."""

    def __init__(self, features, **kwargs):
        super().__init__()

        # Faster way to deep flatten a list of lists compared to list comprehension
        self.feature_dataset = list(chain.from_iterable(features))
        self.trainset = self.feature_dataset
        self.testset = []

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, item):
        return self.trainset[item]
