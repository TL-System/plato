"""
Samples all the data from a dataset. Applicable in cases where the dataset comes from
local sources only. Used by the Federated EMNIST dataset and the MistNet server.
"""
import random

from plato.samplers import base
from plato.config import Config


class Sampler(base.Sampler):
    """Create a data sampler that samples all the data in the dataset.
    Used by the MistNet server.
    """

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
        if hasattr(Config().trainer, "use_mindspore"):
            return list(self.data_samples)
        elif hasattr(Config().trainer, "use_tensorflow"):
            return list(self.data_samples)
        else:
            import torch

            gen = torch.Generator()
            gen.manual_seed(self.random_seed)
            return torch.utils.data.SubsetRandomSampler(
                self.data_samples, generator=gen
            )

    def num_samples(self):
        """Returns the length of the dataset after sampling."""
        return len(self.data_samples)
