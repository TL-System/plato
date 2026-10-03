"""
Samples data from a dataset, biased across labels according to the Dirichlet distribution.
"""

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler

from plato.config import Config
from plato.samplers import base


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across labels according to the Dirichlet distribution."""

    def __init__(self, datasource, client_id, testing):
        super().__init__(client_id)

        # Different clients should have a different bias across the labels & partition size
        self.rng.seed(self.random_seed * int(client_id))

        # Concentration parameter to be used in the Dirichlet distribution
        concentration = (
            Config().data.concentration
            if hasattr(Config().data, "concentration")
            else 1.0
        )

        if testing:
            testset = datasource.get_test_set()
            if testset is None:
                raise RuntimeError(
                    "Dirichlet sampler requires a test dataset when testing is True."
                )
            target_list = getattr(testset, "targets", None)
            if target_list is None:
                raise AttributeError(
                    "Test dataset returned by datasource must expose a 'targets' attribute."
                )
        else:
            # The list of labels (targets) for all the examples
            target_list = datasource.targets()

        class_list = datasource.classes()

        if len(target_list) == 0 or len(class_list) == 0:
            raise ValueError("Dirichlet sampling requires nonempty data and classes.")

        target_proportions = self.rng.dirichlet(
            np.repeat(concentration, len(class_list))
        )

        if np.isnan(np.sum(target_proportions)):
            target_proportions = np.repeat(0, len(class_list))
            target_proportions[self.rng.randint(0, len(class_list))] = 1

        weights = target_proportions[target_list]
        if hasattr(weights, "tolist"):
            weights = weights.tolist()
        self.sample_weights = list(weights)
        sampled_size = Config().data.partition_size

        # Variable partition size across clients
        if hasattr(Config().data, "partition_distribution"):
            dist = Config().data.partition_distribution

            if dist.distribution.lower() == "uniform":
                sampled_size *= self.rng.uniform(dist.low, dist.high)

            if dist.distribution.lower() == "normal":
                sampled_size *= self.rng.normal(dist.mean, dist.high)

            sampled_size = int(sampled_size)

        if not 0 < sampled_size <= len(self.sample_weights):
            raise ValueError(
                "Dirichlet partition size must be positive and no larger than "
                f"the dataset ({len(self.sample_weights)}); got {sampled_size}."
            )
        self.sampled_size = sampled_size

    def num_samples(self) -> int:
        """Return the fixed count realized at construction, without drawing RNG."""
        return self.sampled_size

    def get(self):
        """Obtains an instance of the sampler."""
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)

        # Samples without replacement using the sample weights
        subset_indices = list(
            WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=self.num_samples(),
                replacement=False,
                generator=gen,
            )
        )

        return SubsetRandomSampler(subset_indices, generator=gen)
