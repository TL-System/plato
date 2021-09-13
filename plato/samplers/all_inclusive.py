"""
Samples all the data from a dataset. Applicable in cases where the dataset comes from
local sources only, and used by the MistNet server.
"""
import random
from plato.samplers import base


class Sampler(base.Sampler):
    """Create a data sampler that samples all the data in the dataset.
       Used by the MistNet server.
    """
    def __init__(self, dataset):
        super().__init__()
        self.all_inclusive = range(len(dataset))

    def get(self):
        return random.shuffle(self.all_inclusive)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return len(self.all_inclusive)
