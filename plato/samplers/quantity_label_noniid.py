#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Samples data from a dataset, biased across labels according to the Dirichlet distribution.
This is the Quantity-based label imbalance.
An advantage of current sampler approach is that we can flexibly change the imbalance level by varying the
concentration parameter 𝛽. If 𝛽 is set to a smaller value, then the partition is more unbalanced.
"""
import random
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from plato.config import Config

from plato.samplers import base
'''
Label Distribution Skew:

This sampler is one type of label distribution skew, that is:

    Quantity-based label imbalance: each party owns data samples of a fixed number of labels.

'''


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across labels according to the Dirichlet distribution."""
    def __init__(self, datasource, client_id):
        super().__init__()
        self.client_id = client_id

        # Different clients should have a different bias across the labels
        np.random.seed(self.random_seed * int(client_id))

        self.per_client_classes_size = Config().data.per_client_classes_size

        # Concentration parameter to be used in the Dirichlet distribution
        concentration = Config().data.concentration if hasattr(
            Config().data, 'concentration') else 1.0

        # The list of labels (targets) for all the examples
        target_list = datasource.targets()
        self.class_list = datasource.classes()

        target_proportions = np.random.dirichlet(
            np.repeat(concentration, len(class_list)))

        if np.isnan(np.sum(target_proportions)):
            target_proportions = np.repeat(0, len(class_list))
            target_proportions[random.randint(0, len(class_list) - 1)] = 1

        self.sample_weights = target_proportions[target_list]

    def quantity_label_skew(self, dataset_labels, dataset_classes, num_clients,
                            per_client_classes_size):

        clients_dataidx_map = None

        # each client contains the fully classes
        if per_client_classes_size == len(dataset_classes):
            clients_dataidx_map = self.fully_classes_assigned(
                dataset_labels, dataset_classes, num_clients,
                per_client_classes_size)
        else:
            pass

    def fully_classes_assigned(self, dataset_labels, dataset_classes,
                               num_clients, per_client_classes_size):
        clients_dataidx_map = {
            i: np.ndarray(0, dtype=np.int64)
            for i in range(num_clients)
        }
        for class_i in range(dataset_classes):
            idx_k = np.where(dataset_labels == class_i)[0]

            # the samples of each class is evenly assigned to different clients
            split = np.array_split(idx_k, num_clients)
            for j in range(num_clients):
                clients_dataidx_map[j] = np.append(clients_dataidx_map[j],
                                                   split[j])
        return clients_dataidx_map

    def get(self):
        """Obtains an instance of the sampler. """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)

        # Samples without replacement using the sample weights
        return WeightedRandomSampler(weights=self.sample_weights,
                                     num_samples=self.partition_size,
                                     replacement=False,
                                     generator=gen)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return self.partition_size
