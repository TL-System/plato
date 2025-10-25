"""
HeteroFL algorithm trainer.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Callable, cast

import numpy as np
import torch
from heterofl_algorithm import Algorithm as HeteroAlgorithm
from torch.nn import Module

from plato.config import Config
from plato.samplers import all_inclusive
from plato.servers import fedavg
from plato.servers.strategies.aggregation import FedAvgAggregationStrategy


class HeteroFLAggregationStrategy(FedAvgAggregationStrategy):
    """Aggregation strategy that delegates to the HeteroFL algorithm."""

    async def aggregate_weights(  # pylint: disable=unused-argument
        self, updates, baseline_weights, weights_received, context
    ):
        algorithm = getattr(context, "algorithm", None)
        if algorithm is None or not hasattr(algorithm, "aggregation"):
            return None
        return algorithm.aggregation(weights_received)


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
        super().__init__(
            model,
            datasource,
            algorithm,
            trainer,
            aggregation_strategy=HeteroFLAggregationStrategy(),
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
        self.train_model: Module | None = None

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        algorithm = cast(HeteroAlgorithm, self.require_algorithm())
        limitation = self.limitation[self.current_round - 1, client_id - 1]
        model_factory = self.model
        if model_factory is None:
            raise RuntimeError("Server model factory is not configured.")
        rate = algorithm.choose_rate(
            (float(limitation[0]), float(limitation[1])),
            cast(Callable[..., Module], model_factory),
        )
        server_response["rate"] = rate
        return super().customize_server_response(server_response, client_id)

    def weights_aggregated(self, updates):
        super().weights_aggregated(updates)
        # Implement sBN operation.
        datasource = self.require_datasource()
        algorithm = cast(HeteroAlgorithm, self.require_algorithm())
        model_factory = self.model
        if model_factory is None:
            raise RuntimeError("Server model factory is not configured.")
        trainset = datasource.get_train_set()
        trainset_sampler = all_inclusive.Sampler(datasource, testing=False)
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=Config().trainer.batch_size,
            sampler=trainset_sampler.get(),
        )
        dataset_size = len(trainset) if hasattr(trainset, "__len__") else -1
        logging.info(
            "[Server #%d] Running sBN over %d samples.",
            os.getpid(),
            dataset_size,
        )
        test_model = algorithm.stat(model_factory, trainloader)
        logging.info("[Server #%d] sBN pass complete.", os.getpid())
        self.train_model = copy.deepcopy(algorithm.require_model())
        algorithm.model = test_model

    def clients_processed(self) -> None:
        super().clients_processed()
        algorithm = cast(HeteroAlgorithm, self.require_algorithm())
        if self.train_model is not None:
            algorithm.model = self.train_model
