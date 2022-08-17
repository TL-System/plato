"""
Samples data from a MindSpore dataset in an independent and identically distributed fashion.
"""
import numpy as np

import mindspore.dataset as ds
from mindspore.dataset import SubsetRandomSampler

from plato.samplers import base
from plato.config import Config


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a randomly divided partition of the
    dataset."""

    def __init__(self, datasource, client_id=0, testing=False):
        super().__init__()
        self.client_id = client_id

        indices = list(range(datasource.num_train_examples()))
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)

        partition_size = Config().data.partition_size
        total_clients = Config().clients.total_clients
        total_size = partition_size * total_clients

        # add extra samples to make it evenly divisible, if needed
        if len(indices) < total_size:
            indices += indices[: (total_size - len(indices))]
        else:
            indices = indices[:total_size]
        assert len(indices) == total_size

        # Compute the indices of data in the subset for this client
        self.subset_indices = indices[
            (int(self.client_id) - 1) : total_size : total_clients
        ]

    def get(self):
        """Obtains an instance of the sampler."""
        ds.config.set_seed(self.random_seed)
        return SubsetRandomSampler(self.subset_indices)

    def num_samples(self):
        """Returns the length of the dataset after sampling."""
        return len(self.subset_indices)
