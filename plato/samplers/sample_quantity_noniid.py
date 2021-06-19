#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Samples data from a dataset in an independent and identically distributed fashion.
This sampler achieves the sample quantity noniid. Nmber of samples of each class in each party follows Dirichlet
distribution.

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

        # Concentration parameter to be used in the Dirichlet distribution
        concentration = Config().data.concentration if hasattr(
            Config().data, 'concentration') else 1.0

        min_partition_size = Config().data.min_partition_size

        total_clients = Config().clients.total_clients
        total_size = self.dataset_size

        self.subset_indices = self.sample_quantity_skew(
            dataset_indices=indices,
            dataset_size=self.dataset_size,
            min_partition_size=min_partition_size,
            concentration=concentration,
            num_clients=total_clients)[client_id]

    def sample_quantity_skew(self, dataset_indices, dataset_size,
                             min_partition_size, concentration, num_clients):
        min_size = 0
        while min_size < min_partition_size:
            proportions = np.random.dirichlet(
                np.repeat(concentration, num_clients))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * dataset_size)

        proportions = (np.cumsum(proportions) * dataset_size).astype(int)[:-1]
        print("proportions: ", proportions)
        # obtain the assigned subdataset indices for current client
        # net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        clients_assigned_idxs = np.split(dataset_indices, proportions)
        print("clients_assigned_idxs: ", clients_assigned_idxs)
        return clients_assigned_idxs

    def get(self):
        """Obtains an instance of the sampler. """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        return SubsetRandomSampler(self.subset_indices, generator=gen)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return len(self.subset_indices)
