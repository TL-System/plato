"""
Customized Server for PerFedRLNAS.
"""

from __future__ import annotations

import copy
import pickle
import sys
import time
from typing import Optional, cast

import numpy as np
from fednas_algorithm import ServerAlgorithm, SupernetProtocol

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.aggregation import FedAvgAggregationStrategy


class PerFedRlnasAggregationStrategy(FedAvgAggregationStrategy):
    """Aggregation strategy that delegates to the PerFedRLNAS VIT server logic."""

    async def aggregate_weights(
        self, updates, baseline_weights, weights_received, context
    ):
        server = getattr(context, "server", None)
        if server is None or not hasattr(server, "_aggregate_weights"):
            return None
        result = await server._aggregate_weights(
            updates, baseline_weights, weights_received
        )
        if result is not None:
            return result
        return server._current_global_weights()


class Server(fedavg.Server):
    """The PerFedRLNAS server assigns and aggregates global model with different architectures."""

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
            aggregation_strategy=PerFedRlnasAggregationStrategy(),
        )
        self.subnets_config: list[dict | None] = [
            None for _ in range(Config().clients.total_clients)
        ]
        self.neg_ratio: np.ndarray | None = None
        self.process_begin: float | None = None
        self.process_end: float | None = None
        self.model_size = np.zeros(Config().clients.total_clients)

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        algorithm = cast(ServerAlgorithm, self.require_algorithm())
        subnet_config = algorithm.sample_config(server_response)
        self.subnets_config[server_response["id"] - 1] = subnet_config
        server_response["subnet_config"] = subnet_config

        return server_response

    async def _aggregate_weights(self, updates, baseline_weights, weights_received):  # pylint: disable=unused-argument
        """Aggregates weights of models with different architectures."""
        self.process_begin = time.time()
        client_id_list = [update.client_id for update in self.updates]
        num_samples = [update.report.num_samples for update in self.updates]
        algorithm = cast(ServerAlgorithm, self.require_algorithm())
        self.neg_ratio = algorithm.nas_aggregation(
            self.subnets_config, weights_received, client_id_list, num_samples
        )
        for payload, client_id in zip(weights_received, client_id_list):
            payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
            self.model_size[client_id - 1] = payload_size
        return self._current_global_weights()

    def weights_aggregated(self, updates):
        """After weight aggregation, update the architecture parameter alpha."""
        accuracy_list = [update.report.accuracy for update in updates]
        round_time_list = [
            update.report.training_time + update.report.comm_time
            for update in self.updates
        ]
        client_id_list = [update.client_id for update in self.updates]
        subnet_configs = []
        for client_id_ in client_id_list:
            client_id = client_id_ - 1
            subnet_config = self.subnets_config[client_id]
            subnet_configs.append(subnet_config)

        algorithm = cast(ServerAlgorithm, self.require_algorithm())
        supernet = algorithm._require_supernet()
        epoch_index = supernet.extract_index(subnet_configs)
        neg_ratio = (
            self.neg_ratio
            if self.neg_ratio is not None
            else np.zeros(len(round_time_list))
        )
        supernet.step(
            [accuracy_list, round_time_list, neg_ratio],
            epoch_index,
            client_id_list,
        )
        trainer = self.require_trainer()
        trainer.model = supernet
        self.process_end = time.time()

    def save_to_checkpoint(self) -> None:
        save_config = f"{Config().params['model_path']}/subnet_configs.pickle"
        with open(save_config, "wb") as file:
            pickle.dump(self.subnets_config, file)
        save_config = f"{Config().params['model_path']}/baselines.pickle"
        algorithm = cast(ServerAlgorithm, self.require_algorithm())
        supernet = algorithm._require_supernet()
        with open(save_config, "wb") as file:
            pickle.dump(supernet.baseline, file)
        return super().save_to_checkpoint()

    def get_logged_items(self) -> dict:
        logged_items = super().get_logged_items()
        algorithm = cast(ServerAlgorithm, self.require_algorithm())
        acc_info = algorithm.get_baseline_accuracy_info()
        logged_items["clients_accuracy_mean"] = acc_info["mean"]
        logged_items["clients_accuracy_std"] = acc_info["std"]
        logged_items["clients_accuracy_max"] = acc_info["max"]
        logged_items["clients_accuracy_min"] = acc_info["min"]
        if self.process_begin is not None and self.process_end is not None:
            logged_items["server_overhead"] = self.process_end - self.process_begin
        logged_items["model_size"] = np.mean(self.model_size)
        return logged_items

    def _current_global_weights(self):
        """Return a copy of the current global model weights."""
        algorithm = cast(ServerAlgorithm, self.require_algorithm())
        model: Optional[SupernetProtocol] = getattr(algorithm, "model", None)
        if model is None:
            return None
        if hasattr(model, "state_dict"):
            return copy.deepcopy(model.state_dict())
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "state_dict"):
            return copy.deepcopy(inner.state_dict())
        return None
