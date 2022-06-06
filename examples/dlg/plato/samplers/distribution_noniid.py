"""
Samples data from a dataset, biased across labels according to the Dirichlet
distribution and biased across data size according to Dirichlet distribution.

This sampler can introduce the hardest non-IID data scenarios because it contains:

    - Label skewness - equals to the sampler called dirichlet, i.e., dirichlet.py
        The number of classes contained in clients follows the Dirichlet distribution
        that is parameterized by the "label_concentration".
    - Quantity skewness - equals to the sampler called "sample_quantity_noniid.py".
	    The local dataset sizes of clients follow the  Dirichlet distribution that is
        parameterized by the "client_quantity_concentration".

    For example,
        1. Setting label_concentration = 0.1 will induce extreme label unbalance between clients.
        When there are ten classes, each client only contains sufficient samples from one class.
                classes 1       2       3 ...   8     9
                client1 100     8       9       3     7
                client2 4      108      7       9     6
                ...
                clientN 3       10      11      99    2
        2. Setting client_quantity_concentration = 0.1 will induce extreme data scale
         unbalance between clients.
        The sample sizes of clients follow the Dirichlet distribution.
                classes 1       2       3 ...   8     9
                client1 5       6       7       5     8
                client2 50      45      67      49    56
                ...
                clientN 6       7      11      10     7
        3. Then, this sampler introduces the above two unbalance conditions simultaneously.
                classes 1       2       3 ...   8     9
                client1 60      66      380     45    38
                client2 90      5       3       6     8
                ...
                clientN 1       50      1      1      1
"""
import numpy as np
import torch

from torch.utils.data import WeightedRandomSampler

from plato.config import Config
from plato.samplers import base
from plato.samplers import sampler_utils


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across labels according to the Dirichlet distribution
    and biased partition size."""
    def __init__(self, datasource, client_id, testing):
        super().__init__()
        self.client_id = client_id

        # set the random seed based on client id
        np.random.seed(self.random_seed * int(client_id))

        # obtain the dataset information
        if testing:
            target_list = datasource.get_test_set().targets
        else:
            # the list of labels (targets) for all the examples
            target_list = datasource.get_train_set().targets

        class_list = datasource.classes()
        total_data_size = len(target_list)

        # obtain the configuration
        min_partition_size = Config().data.min_partition_size if hasattr(
            Config().data, 'min_partition_size') else 100
        total_clients = Config().clients.total_clients

        client_quantity_concentration = Config(
        ).data.client_quantity_concentration if hasattr(
            Config().data, 'client_quantity_concentration') else 1.0

        label_concentration = Config().data.label_concentration if hasattr(
            Config().data, 'label_concentration') else 1.0

        self.client_partition = sampler_utils.create_dirichlet_skew(
            total_size=total_data_size,
            concentration=client_quantity_concentration,
            min_partition_size=None,
            number_partitions=total_clients)[client_id]

        self.client_partition_size = int(total_data_size *
                                         self.client_partition)
        if self.client_partition_size < min_partition_size:
            self.client_partition_size = min_partition_size

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

    def data_size(self):
        """Returns the length of the dataset after sampling. """
        return self.client_partition_size

    def trainset_size(self):
        """ Returns the length of the train dataset """
        return self.data_size()

    def get_sampler_condition(self):
        """ Obtain the label ratio and the sampler configuration """
        return self.client_partition, self.client_label_proportions
