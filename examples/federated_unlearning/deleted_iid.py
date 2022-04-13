"""
A customized iid sampler for federated unlearning.

Reference: Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining." in Proc. INFOCOM, 2022 https://arxiv.org/abs/2203.07320

"""
import numpy as np

from plato.config import Config
from plato.samplers import iid


class Sampler(iid.Sampler):
    """Create a data sampler for each client to use a randomly divided partition of the
    dataset with delete_data_ratio."""

    def __init__(self, datasource, client_id, testing, need_delete=False):
        super().__init__(datasource, client_id, testing)
        if testing:
            dataset = datasource.get_test_set()
        else:
            dataset = datasource.get_train_set()

        self.dataset_size = len(dataset)
        indices = list(range(self.dataset_size))
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)

        partition_size = Config().data.partition_size
        total_clients = Config().clients.total_clients
        total_size = partition_size * total_clients
        delete_data_ratio = Config().clients.delete_data_ratio

        # add extra samples to make it evenly divisible, if needed
        if len(indices) < total_size:
            while len(indices) < total_size:
                indices += indices[:(total_size - len(indices))]
        else:
            indices = indices[:total_size]
        assert len(indices) == total_size

        # Compute the indices of data in the subset for this client
        self.subset_indices = indices[(int(client_id) -
                                       1):total_size:total_clients]

        if (need_delete):
            num_subset_length = len(self.subset_indices)
            delelte_subset_length = num_subset_length * delete_data_ratio
            delete_index = np.random.choice(range(num_subset_length),
                                            delelte_subset_length,
                                            replace=False)

            self.subset_indices = set(
                np.delete(list(self.subset_indices), delete_index))
