"""
Base class for sampling data so that a dataset can be divided across the clients.
"""
import os
from abc import abstractmethod

from plato.config import Config
from plato.samplers import base

class Sampler(base.Sampler):
    """Base class for data samplers so that the dataset is divided into
    partitions across the clients."""
    def __init__(self, datasource, client_id):
        super().__init__()
        dataset = datasource.get_train_set()
        self.all_inclusive = range(len(dataset))

    @abstractmethod
    def get(self):
        """Obtains an instance of the sampler. """
        return None

    def trainset_size(self):
        return len(self.all_inclusive)
