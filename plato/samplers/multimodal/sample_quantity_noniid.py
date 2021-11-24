"""
Samples data from a dataset in an independent and identically distributed fashion.
This sampler achieves the sample quantity noniid. Nmber of samples of each class
 in each party follows Dirichlet distribution.

For each client, it will contain all classes while the number of samples in
 each class is the almost the same.

"""

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

from plato.config import Config
from plato.samplers import base
from plato.samplers.multimodal import sampler_utils


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

        # The list of labels (targets) for all the examples
        self.targets_list = datasource.targets()
        # classes_text_list = datasource.classes()
        # classes_id_list = list(range(len(classes_text_list)))

        # Concentration parameter to be used in the Dirichlet distribution
        concentration = Config().data.concentration if hasattr(
            Config().data, 'concentration') else 1.0

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
            number_partitions=num_clients)

        proportions = (np.cumsum(proportions) * dataset_size).astype(int)[:-1]

        # obtain the assigned subdataset indices for current client
        # net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        clients_assigned_idxs = np.split(dataset_indices, proportions)

        return clients_assigned_idxs

    def get(self):
        """Obtains an instance of the sampler. """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        return SubsetRandomSampler(self.subset_indices, generator=gen)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return len(self.subset_indices)

    def get_trainset_condition(self):
        """ Get the detailed info of the trainset """
        targets_array = np.array(self.targets_list)
        client_sampled_subset_labels = targets_array[self.subset_indices]
        unique, counts = np.unique(client_sampled_subset_labels,
                                   return_counts=True)
        return np.asarray((unique, counts)).T
