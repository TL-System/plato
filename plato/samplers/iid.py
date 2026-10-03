"""
Samples data from a dataset in an independent and identically distributed fashion.
"""

import torch
from torch.utils.data import SubsetRandomSampler

from plato.config import Config
from plato.samplers import base, sampler_utils


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a randomly divided partition of the
    dataset."""

    def __init__(self, datasource, client_id, testing):
        super().__init__(client_id)

        if testing:
            dataset = datasource.get_test_set()
        else:
            dataset = datasource.get_train_set()

        self.dataset_size = len(dataset)
        if self.dataset_size == 0:
            raise ValueError(
                "IID sampling requires a nonempty dataset; got empty data."
            )
        indices = list(range(self.dataset_size))
        self.rng.shuffle(indices)

        partition_size = Config().data.partition_size
        total_clients = Config().clients.total_clients
        total_size = partition_size * total_clients
        if partition_size < 0 or total_clients <= 0:
            raise ValueError(
                "partition_size must be nonnegative and total_clients positive."
            )

        # add extra samples to make it evenly divisible, if needed
        indices = sampler_utils.extend_indices(indices, total_size)

        # Compute the indices of data in the subset for this client
        self.subset_indices = indices[(int(client_id) - 1) : total_size : total_clients]

    def get(self):
        """Obtains an instance of the sampler."""
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        return SubsetRandomSampler(self.subset_indices, generator=gen)

    def num_samples(self):
        """Returns the length of the dataset after sampling."""
        return len(self.subset_indices)
