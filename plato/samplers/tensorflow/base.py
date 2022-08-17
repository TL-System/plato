"""
Base class for sampling data so that a dataset can be divided across the clients.
"""
from abc import abstractmethod

from plato.samplers import base


class Sampler(base.Sampler):
    """Base class for data samplers so that the dataset is divided into
    partitions across the clients."""

    def __init__(self, datasource, client_id=0, testing=False):
        super().__init__()
        self.client_id = client_id

        if testing:
            self.all_inclusive = range(datasource.num_test_examples())
        else:
            self.all_inclusive = range(datasource.num_train_examples())

    @abstractmethod
    def get(self):
        """Obtains an instance of the sampler."""
        return None

    def num_samples(self):
        return len(self.all_inclusive)
