"""
Samples all data from a dataset indexed by client_id.
"""
import numpy as np
from torch.utils.data import SequentialSampler

from plato.samplers import base


class Sampler(base.Sampler):
    """Create a data sampler for each client to sample all local data in a shuffled order."""
    def __init__(self, datasource, client_id):
        super().__init__()
        self.client_id = client_id
        dataset = datasource.get_train_set()
        self.dataset_size = len(dataset[client_id])
        indices = list(range(self.dataset_size))
        np.random.shuffle(indices)

        # Compute the indices of data in the subset for this client
        self.subset_indices = indices

    def get(self):
        """Obtains an instance of the sampler. """
        return SequentialSampler(self.subset_indices)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return len(self.subset_indices)
