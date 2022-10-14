"""
A sampler for orthogonal cross-silo federated learning.
Each insitution's clients have data of different classes.
"""
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler
from plato.config import Config

from plato.samplers import base


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the dataset.
    A client only has data of certain classes."""

    def __init__(self, datasource, client_id, testing):
        super().__init__()

        # Different clients should have a different bias across the labels
        np.random.seed(self.random_seed * int(client_id))

        self.partition_size = Config().data.partition_size

        if testing:
            target_list = datasource.get_test_set().targets
        else:
            # The list of labels (targets) for all the examples
            target_list = datasource.targets()
        class_list = datasource.classes()

        max_client_id = int(Config().clients.total_clients)

        if client_id > max_client_id:
            # This client is an edge server
            institution_id = client_id - 1 - max_client_id
        else:
            institution_id = (client_id - 1) % int(Config().algorithm.total_silos)

        if hasattr(Config().data, "institution_class_ids"):
            institution_class_ids = Config().data.institution_class_ids
            class_ids = [x.strip() for x in institution_class_ids.split(";")][
                institution_id
            ]
            class_id_list = [int(x.strip()) for x in class_ids.split(",")]
        else:
            class_ids = np.array_split(
                [i for i in range(len(class_list))], Config().algorithm.total_silos
            )[institution_id]
            class_id_list = class_ids.tolist()

        if (
            hasattr(Config().data, "label_distribution")
            and Config().data.label_distribution == "noniid"
        ):
            # Concentration parameter to be used in the Dirichlet distribution
            concentration = (
                Config().data.concentration
                if hasattr(Config().data, "concentration")
                else 1.0
            )

            class_proportions = np.random.dirichlet(
                np.repeat(concentration, len(class_id_list))
            )

        else:
            class_proportions = [
                1.0 / len(class_id_list) for i in range(len(class_id_list))
            ]

        target_proportions = [0 for i in range(len(class_list))]
        for index, class_id in enumerate(class_id_list):
            target_proportions[class_id] = class_proportions[index]
        target_proportions = np.asarray(target_proportions)

        self.sample_weights = target_proportions[target_list]

    def get(self):
        """Obtains an instance of the sampler."""
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)

        # Samples without replacement using the sample weights
        subset_indices = list(
            WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=self.partition_size,
                replacement=False,
                generator=gen,
            )
        )

        return SubsetRandomSampler(subset_indices, generator=gen)

    def num_samples(self):
        """Returns the length of the dataset after sampling."""
        return self.partition_size
