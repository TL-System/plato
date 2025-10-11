"""
A customized i.i.d. sampler for federated unlearning.

Federated unlearning allows clients to proactively erase their data from a trained model. The model
will be retrained from scratch during the unlearning process.

If the AdaHessian optimizer is used, it will reflect what the following paper proposed:

Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid
Retraining," in Proc. INFOCOM, 2022.

Reference: https://arxiv.org/abs/2203.07320
"""

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

from plato.config import Config
from plato.samplers import iid


class Sampler(iid.Sampler):
    """Create a data sampler for each client to use a randomly divided partition of the
    dataset with a particular ratio of data to be deleted."""

    def __init__(self, datasource, client_id, testing):
        super().__init__(datasource, client_id, testing)
        if testing:
            dataset = datasource.get_test_set()
        else:
            dataset = datasource.get_train_set()

        self.dataset_size = len(dataset)
        indices = list(range(self.dataset_size))
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)

        partition_size = Config().data.partition_size
        total_clients = Config().clients.total_clients
        total_size = partition_size * total_clients
        deleted_data_ratio = Config().clients.deleted_data_ratio

        # add extra samples to make it evenly divisible, if needed
        if len(indices) < total_size:
            while len(indices) < total_size:
                indices += indices[: (total_size - len(indices))]
        else:
            indices = indices[:total_size]
        assert len(indices) == total_size

        # Compute the indices of data in the subset for this client
        self.subset_indices = indices[(int(client_id) - 1) : total_size : total_clients]
        subset_length = int(len(self.subset_indices))
        deleted_subset_length = int(subset_length * deleted_data_ratio)
        deleted_index = np.random.choice(
            range(subset_length), deleted_subset_length, replace=False
        )
        self.deleted_subset_indices = [self.subset_indices[i] for i in deleted_index]
        self.subset_indices = list(np.delete(self.subset_indices, deleted_index))

    def get_negative(self):
        """Obtains an instance of the sampler for the data to be deleted (unlearned)."""
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        return SubsetRandomSampler(self.deleted_subset_indices, generator=gen)
