"""
Samples data from a dataset, biased across labels according to the Dirichlet distribution.
"""
import random
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from plato.config import Config

from plato.samplers import base


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across labels according to the Dirichlet distribution."""
    def __init__(self, datasource, client_id, testing):
        super().__init__()

        # Different clients should have a different bias across the labels
        np.random.seed(self.random_seed * int(client_id))

        self.partition_size = Config().data.partition_size

        # Concentration parameter to be used in the Dirichlet distribution
        concentration = Config().data.concentration if hasattr(
            Config().data, 'concentration') else 1.0

        if testing:
            target_list = datasource.get_test_set().targets
        else:
            # The list of labels (targets) for all the examples
            target_list = datasource.targets()

        class_list = datasource.classes()

        target_proportions = np.random.dirichlet(
            np.repeat(concentration, len(class_list)))

        if np.isnan(np.sum(target_proportions)):
            target_proportions = np.repeat(0, len(class_list))
            target_proportions[random.randint(0, len(class_list) - 1)] = 1

        self.sample_weights = target_proportions[target_list]

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
