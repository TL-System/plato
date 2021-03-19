"""
Samples data from a dataset, biased across labels according to the Dirichlet distribution.
"""
import torch
import random
import numpy as np

from samplers import base
from config import Config
from torch.utils.data import WeightedRandomSampler


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across labels according to the Dirichlet distribution."""
    def __init__(self, datasource, client_id):
        super().__init__(datasource)
        self.client_id = client_id

        # Different clients should have a different bias across the labels
        np.random.seed(self.random_seed * int(client_id))

        self.partition_size = Config().data.partition_size
        concentration = Config().data.concentration if hasattr(
            Config().data, 'concentration') else 1.0

        # The list of labels (targets) for all the examples
        target_list = datasource.get_train_set().targets
        class_list = datasource.classes()

        target_proportions = np.random.dirichlet(
            np.repeat(concentration, len(class_list)))

        self.sample_weights = target_proportions[target_list]

    def get(self):
        """Obtain an instance of the sampler. """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)

        # Samples without replacement using the sample weights
        return WeightedRandomSampler(self.sample_weights,
                                     self.partition_size,
                                     replacement=False,
                                     generator=gen)

    def trainset_size(self):
        return self.partition_size
