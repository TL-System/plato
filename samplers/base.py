"""
Base class for sampling data so that a dataset can be divided across the clients.
"""
import os
from abc import abstractmethod

from config import Config


class Sampler:
    """Base class for data samplers so that the dataset is divided into
    partitions across the clients."""
    def __init__(self, datasource):
        """Get data from the data source."""
        dataset = datasource.get_train_set()
        self.dataset_size = len(dataset)

        if hasattr(Config().data, 'random_seed'):
            # Keeping random seed the same across the clients
            # so that the experiments are reproducible
            self.random_seed = Config().data.random_seed
        else:
            # The random seed will be different across different
            # runs if it is not provided.
            self.random_seed = os.getpid()

    @abstractmethod
    def get(self):
        """Obtains an instance of the sampler. """

    @abstractmethod
    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
