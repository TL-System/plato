"""
Samples data from a dataset as clients' local datasets.
Some are biased across labels according to the Dirichlet distribution,
while some are in an independent and identically distributed fashion.
"""
import numpy as np
from plato.config import Config
from plato.samplers import dirichlet


class Sampler(dirichlet.Sampler):
    """Create a data sampler for each client to use a divided partition of the dataset,
    either biased across labels according to the Dirichlet distribution, or in an iid fashion."""
    def __init__(self, datasource, client_id, testing):
        super().__init__(datasource, client_id, testing)

        assert hasattr(Config().data, 'non_iid_clients')
        non_iid_clients = Config().data.non_iid_clients

        if isinstance(non_iid_clients, int):
            # If only one client's dataset is non-iid
            self.non_iid_clients_list = [int(non_iid_clients)]
        else:
            self.non_iid_clients_list = [
                int(x.strip()) for x in non_iid_clients.split(',')
            ]

        if int(client_id) not in self.non_iid_clients_list:
            if testing:
                target_list = datasource.get_test_set().targets
            else:
                target_list = datasource.targets()
            class_list = datasource.classes()
            self.sample_weights = np.array([
                1 / len(class_list) for _ in range(len(class_list))
            ])[target_list]

            # Different iid clients should have a different random seed for Generator
            self.random_seed = self.random_seed * int(client_id)
