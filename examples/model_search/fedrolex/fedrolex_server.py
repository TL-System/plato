"""
FedRolexFL algorithm trainer.
"""

import numpy as np

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.aggregation import FedAvgAggregationStrategy


class FedRolexAggregationStrategy(FedAvgAggregationStrategy):
    """Aggregation strategy that delegates to the FedRolex algorithm."""

    async def aggregate_weights(
        self, updates, baseline_weights, weights_received, context  # pylint: disable=unused-argument
    ):
        algorithm = getattr(context, "algorithm", None)
        if algorithm is None or not hasattr(algorithm, "aggregation"):
            return None
        return algorithm.aggregation(weights_received)


class Server(fedavg.Server):
    """A federated learning server using the FedRolexFL algorithm."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(
            model,
            datasource,
            algorithm,
            trainer,
            aggregation_strategy=FedRolexAggregationStrategy(),
        )
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

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        rate = self.algorithm.choose_rate(
            self.limitation[self.current_round - 1, client_id - 1], self.model
        )
        server_response["rate"] = rate
        self.rates[client_id - 1] = rate
        return super().customize_server_response(server_response, client_id)

    def weights_aggregated(self, updates):
        super().weights_aggregated(updates)
        self.algorithm.sort_channels()
