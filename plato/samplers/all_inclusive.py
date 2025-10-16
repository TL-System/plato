"""
Samples all the data from a dataset. Applicable in cases where the dataset comes from
local sources only. Used by the Federated EMNIST dataset.
"""

import random

from plato.config import Config
from plato.samplers import base


class Sampler(base.Sampler):
    """Create a data sampler that samples all the data in the dataset."""

    def __init__(self, datasource, client_id=0, testing=False):
        super().__init__()
        self.client_id = client_id

        if testing:
            all_inclusive = range(len(datasource.get_test_set()))
            if hasattr(Config().data, "testset_size"):
                self.data_samples = random.sample(
                    all_inclusive, Config().data.testset_size
                )
            else:
                self.data_samples = all_inclusive
        else:
            self.data_samples = range(len(datasource.get_train_set()))

    def get(self):
        import torch

        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        return torch.utils.data.SubsetRandomSampler(self.data_samples, generator=gen)

    def num_samples(self):
        """Returns the length of the dataset after sampling."""
        return len(self.data_samples)
