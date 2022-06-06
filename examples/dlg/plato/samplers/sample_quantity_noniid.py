'''
Samples data from a dataset, biased across quantity size of clients.

This sampler implements one type of sample distribution skew called:

    Quantity skewness:
        The local dataset sizes of clients follow the  Dirichlet distribution that is
        parameterized by the "client_quantity_concentration".

        Within each client, sample sizes of different classes are the same.

    For Example:
        Setting client_quantity_concentration = 0.1 will induce extreme data scale
         unbalance between clients.
        The sample sizes of clients follow the Dirichlet distribution.
                classes 1       2       3 ...   8     9
                client1 5       6       7       5     8
                client2 50      45      67      49    56
                ...
                clientN 6       7      11      10     7

'''

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

from plato.config import Config
from plato.samplers import base
from plato.samplers import sampler_utils


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across partition size."""
    def __init__(self, datasource, client_id, testing):
        super().__init__()

        self.client_id = client_id

        np.random.seed(self.random_seed)

        # obtain the dataset information
        if testing:
            dataset = datasource.get_test_set()
        else:
            dataset = datasource.get_train_set()

        # The list of labels (targets) for all the examples
        self.targets_list = datasource.targets

        self.dataset_size = len(dataset)

        indices = list(range(self.dataset_size))

        np.random.shuffle(indices)

        # Concentration parameter to be used in the Dirichlet distribution
        concentration = Config().data.client_quantity_concentration if hasattr(
            Config().data, 'client_quantity_concentration') else 1.0

        min_partition_size = Config().data.min_partition_size
        total_clients = Config().clients.total_clients

        self.subset_indices = self.sample_quantity_skew(
            dataset_indices=indices,
            dataset_size=self.dataset_size,
            min_partition_size=min_partition_size,
            concentration=concentration,
            num_clients=total_clients)[client_id]

    def sample_quantity_skew(self, dataset_indices, dataset_size,
                             min_partition_size, concentration, num_clients):
        """ Create the quantity-based sample skewness """
        proportions = sampler_utils.create_dirichlet_skew(
            total_size=dataset_size,
            concentration=concentration,
            min_partition_size=min_partition_size,
            number_partitions=num_clients,
            is_extend_total_size=True)

        proportions_range = (np.cumsum(proportions) *
                             dataset_size).astype(int)[:-1]

        required_total_size = proportions_range[-1]
        extended_dataset_indices = sampler_utils.extend_indices(
            indices=dataset_indices, required_total_size=required_total_size)

        # obtain the assigned subdataset indices for current client
        clients_assigned_idxs = np.split(extended_dataset_indices,
                                         proportions_range)

        return clients_assigned_idxs

    def get(self):
        """Obtains an instance of the sampler. """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        return SubsetRandomSampler(self.subset_indices, generator=gen)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return len(self.subset_indices)

    def get_sampled_data_condition(self):
        """ Get the detailed info of the trainset """
        targets_array = np.array(self.targets_list)
        client_sampled_subset_labels = targets_array[self.subset_indices]
        unique, counts = np.unique(client_sampled_subset_labels,
                                   return_counts=True)
        return np.asarray((unique, counts)).T
