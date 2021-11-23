"""
Samples data from a dataset, biased across labels according to the Dirichlet
distribution and quantity skew.

This sampler is the most hard noniid, because it contains:

    - the label noniid according to the Dirichlet distribution
    - the unbalance between numbers of samples in classes assigned to one client.
        It is achieved according to the Dirichlet distribution
"""
import numpy as np
import torch

from torch.utils.data import WeightedRandomSampler

from plato.config import Config
from plato.samplers import base
from plato.samplers.multimodal import sampler_utils


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across labels according to the Dirichlet distribution
    and biased partition size."""
    def __init__(self, datasource, client_id):
        super().__init__()
        self.client_id = client_id

        # Different clients should have a different bias across the labels
        np.random.seed(self.random_seed * int(client_id))

        # Obtain the dataset information
        dataset = datasource.get_train_set()
        total_data_size = len(dataset)

        # Obtain the configuration
        min_partition_size = Config().data.min_partition_size
        total_clients = Config().clients.total_clients

        # Concentration parameter to be used in the Dirichlet distribution
        ## control the label noniid distribution among clients
        client_quantity_concentration = Config(
        ).data.client_quantity_concentration if hasattr(
            Config().data, 'client_quantity_concentration') else 1.0
        ## control the number of samples of labels in one client
        label_concentration = Config().data.label_concentration if hasattr(
            Config().data, 'label_concentration') else 1.0

        # The list of labels (targets) for all the examples
        target_list = datasource.targets()
        class_list = datasource.classes()

        self.client_partition = sampler_utils.create_dirichlet_skew(
            total_size=total_data_size,
            concentration=client_quantity_concentration,
            min_partition_size=min_partition_size,
            number_partitions=total_clients)[client_id]

        self.client_partition_size = int(total_data_size *
                                         self.client_partition)

        self.client_label_proportions = sampler_utils.create_dirichlet_skew(
            total_size=len(class_list),
            concentration=label_concentration,
            min_partition_size=None,
            number_partitions=len(class_list))

        self.sample_weights = self.client_label_proportions[target_list]

    def get(self):
        """Obtains an instance of the sampler. """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)

        # Samples without replacement using the sample weights
        return WeightedRandomSampler(weights=self.sample_weights,
                                     num_samples=self.client_partition_size,
                                     replacement=False,
                                     generator=gen)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return self.client_partition_size

    def get_trainset_condition(self):
        """ Obtain the label ratio and the sampler configuration """
        return self.client_label_proportions, self.sample_weights
