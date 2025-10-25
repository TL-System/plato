"""
The server for paper system-heterogenous federated learning through architecture search.
"""

import numpy as np

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.aggregation import FedAvgAggregationStrategy


class SysHeteroFLAggregationStrategy(FedAvgAggregationStrategy):
    """Aggregation strategy that delegates to the SysHeteroFL algorithm."""

    async def aggregate_weights(
        self, updates, baseline_weights, weights_received, context  # pylint: disable=unused-argument
    ):
        algorithm = getattr(context, "algorithm", None)
        if algorithm is None or not hasattr(algorithm, "aggregation"):
            return None
        return algorithm.aggregation(weights_received)


class Server(fedavg.Server):
    """A federated learning server using the ElasticArch algorithm."""

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
            aggregation_strategy=SysHeteroFLAggregationStrategy(),
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
        server_response["config"] = self.algorithm.choose_config(
            self.limitation[self.current_round - 1, client_id - 1]
        )
        self.trainer.biggest_net_config = self.algorithm.biggest_net
        return super().customize_server_response(server_response, client_id)

    def clients_processed(self) -> None:
        super().clients_processed()
        if (
            hasattr(Config().parameters, "distillation")
            and hasattr(Config().parameters.distillation, "activate")
            and Config().parameters.distillation.activate
        ):
            self.algorithm.distillation()

    def training_will_start(self) -> None:
        super().training_will_start()
        self.algorithm.initialize_arch_map(self.model)
