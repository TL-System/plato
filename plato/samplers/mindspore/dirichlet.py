"""
Samples data from a dataset, biased across labels according to the Dirichlet distribution.
"""
import numpy as np

import mindspore.dataset as ds
from mindspore.dataset import WeightedRandomSampler, SubsetRandomSampler

from plato.samplers import base
from plato.config import Config


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across labels according to the Dirichlet distribution."""

    def __init__(self, datasource, client_id=0, testing=False):
        super().__init__(datasource)
        self.client_id = client_id

        # Different clients should have a different bias across the labels
        np.random.seed(self.random_seed * int(client_id))

        self.partition_size = Config().data.partition_size

        # Concentration parameter to be used in the Dirichlet distribution
        concentration = (
            Config().data.concentration
            if hasattr(Config().data, "concentration")
            else 1.0
        )

        # The list of labels (targets) for all the examples
        target_list = datasource.targets()
        class_list = datasource.classes()

        target_proportions = np.random.dirichlet(
            np.repeat(concentration, len(class_list))
        )

        self.sample_weights = target_proportions[target_list]

    def get(self):
        """Obtains an instance of the sampler."""
        ds.config.set_seed(self.random_seed)

        # Samples without replacement using the sample weights
        subset_indices = list(
            WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=self.partition_size,
                replacement=False,
            )
        )

        return SubsetRandomSampler(subset_indices)

    def num_samples(self):
        """Returns the length of the dataset after sampling."""
        return self.partition_size
