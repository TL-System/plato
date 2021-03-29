"""
Samples data from a dataset in an independent and identically distributed fashion.
"""
import torch
import numpy as np

from samplers import iid
from config import Config
from torch.utils.data import WeightedRandomSampler


class Sampler(iid.Sampler):
    """Create a data sampler for each client to use a randomly divided partition of the
    dataset."""
    def __init__(self, datasource, client_id):
        super().__init__(datasource, client_id)

        non_iid_clients = Config().data.non_iid_clients
        if isinstance(non_iid_clients, int):
            # if only one client's dataset is non-iid
            self.non_iid_clients_list = [str(non_iid_clients)]
        else:
            self.non_iid_clients_list = [
                x.strip() for x in non_iid_clients.split(',')
            ]

        # Use Dirichlet distribution for non-iid clients
        if self.client_id in self.non_iid_clients_list:
            # Different clients should have a different bias across the labels
            np.random.seed(self.random_seed * int(client_id))

            self.partition_size = Config().data.partition_size

            # Concentration parameter to be used in the Dirichlet distribution
            concentration = Config().data.concentration if hasattr(
                Config().data, 'concentration') else 1.0

            # The list of labels (targets) for all the examples
            target_list = datasource.targets()
            class_list = datasource.classes()

            target_proportions = np.random.dirichlet(
                np.repeat(concentration, len(class_list)))

            self.sample_weights = target_proportions[target_list]

    def get(self):
        """Obtains an instance of the sampler. """
        if self.client_id in self.non_iid_clients_list:
            gen = torch.Generator()
            gen.manual_seed(self.random_seed)

            # Samples without replacement using the sample weights
            return WeightedRandomSampler(weights=self.sample_weights,
                                         num_samples=self.partition_size,
                                         replacement=False,
                                         generator=gen)
        else:
            super().get()
