"""
Samples data from a dataset in an independent and identically distributed fashion.
"""
import numpy as np
import torch
from plato.config import Config
from torch.utils.data import SubsetRandomSampler

from plato.samplers import base


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a randomly divided partition of the
    dataset."""
    def __init__(self, datasource, client_id):
        super().__init__()
        self.client_id = client_id
        dataset = datasource.get_train_set()
        self.dataset_size = len(dataset)
        indices = list(range(self.dataset_size))
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)

        partition_size = Config().data.partition_size
        total_clients = Config().clients.total_clients
        total_size = partition_size * total_clients

        # add extra samples to make it evenly divisible, if needed
        if len(indices) < total_size:
            while len(indices) < total_size:
                indices += indices[:(total_size - len(indices))]
        else:
            indices = indices[:total_size]
        assert len(indices) == total_size

        # Compute the indices of data in the subset for this client
        self.subset_indices = indices[(int(self.client_id) -
                                       1):total_size:total_clients]

    def get(self):
        """Obtains an instance of the sampler. """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        return SubsetRandomSampler(self.subset_indices, generator=gen)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return len(self.subset_indices)
