"""
HeteroFL algorithm trainer.
"""

import copy

import numpy as np

from plato.config import Config
from plato.samplers import all_inclusive
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the HeteroFL algorithm."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(model, datasource, algorithm, trainer)
        self.rates = [None for _ in range(Config().clients.total_clients)]
        self.limitation = np.zeros(
            (Config().trainer.rounds, Config().clients.total_clients, 2)
        )
        if (
            hasattr(Config().parameters.limitation, "activated")
            and Config().parameters.limitation.activated
        ):
            limitation = Config().parameters.limitation
            self.limitation[:, :, 0] = np.random.uniform(
                limitation.min_size,
                limitation.max_size,
                (Config().trainer.rounds, Config().clients.total_clients),
            )
            self.limitation[:, :, 1] = np.random.uniform(
                limitation.min_flops,
                limitation.max_flops,
                (Config().trainer.rounds, Config().clients.total_clients),
            )
        self.train_model = None

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        server_response["rate"] = self.algorithm.choose_rate(
            self.limitation[self.current_round - 1, client_id - 1], self.model
        )
        return super().customize_server_response(server_response, client_id)

    async def aggregate_weights(self, updates, baseline_weights, weights_received):  # pylint: disable=unused-argument
        """Aggregates weights of models with different architectures."""
        return self.algorithm.aggregation(weights_received)

    def weights_aggregated(self, updates):
        super().weights_aggregated(updates)
        # Implement sBN operation.
        trainset = self.datasource.get_train_set()
        trainset_sampler = all_inclusive.Sampler(self.datasource, testing=False)
        trainloader = self.trainer.get_train_loader(
            Config().trainer.batch_size, trainset, trainset_sampler.get()
        )
        test_model = self.algorithm.stat(self.model, trainloader)
        self.train_model = copy.deepcopy(self.algorithm.model)
        self.algorithm.model = test_model

    def clients_processed(self) -> None:
        super().clients_processed()
        self.algorithm.model = self.train_model
