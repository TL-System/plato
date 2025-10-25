"""
FedRolexFL algorithm trainer.
"""

from __future__ import annotations

from typing import Callable, cast

import numpy as np
from fedrolex_algorithm import Algorithm as FedRolexAlgorithm
from torch.nn import Module

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.aggregation import FedAvgAggregationStrategy


class FedRolexAggregationStrategy(FedAvgAggregationStrategy):
    """Aggregation strategy that delegates to the FedRolex algorithm."""

    async def aggregate_weights(  # pylint: disable=unused-argument
        self, updates, baseline_weights, weights_received, context
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
            hasattr(Config().parameters, "limitation")
            and hasattr(Config().parameters.limitation, "activated")
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
        algorithm = cast(FedRolexAlgorithm, self.require_algorithm())
        limitation = self.limitation[self.current_round - 1, client_id - 1]
        model_factory = self.model
        if model_factory is None:
            raise RuntimeError("Server model factory is not configured.")
        rate = algorithm.choose_rate(
            (float(limitation[0]), float(limitation[1])),
            cast(Callable[..., Module], model_factory),
        )
        server_response["rate"] = rate
        self.rates[client_id - 1] = rate
        return super().customize_server_response(server_response, client_id)

    def weights_aggregated(self, updates):
        super().weights_aggregated(updates)
        algorithm = cast(FedRolexAlgorithm, self.require_algorithm())
        algorithm.sort_channels()
