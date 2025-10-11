"""
Customized Server for PerFedRLNAS.
"""

import pickle
import sys
import time

import numpy as np

from plato.config import Config
from plato.servers import fedavg


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
        super().__init__(model, datasource, algorithm, trainer)
        self.subnets_config = [None for i in range(Config().clients.total_clients)]
        self.neg_ratio = None
        self.process_begin = None
        self.process_end = None
        self.model_size = np.zeros(Config().clients.total_clients)

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        subnet_config = self.algorithm.sample_config(server_response)
        self.subnets_config[server_response["id"] - 1] = subnet_config
        server_response["subnet_config"] = subnet_config

        return server_response

    async def aggregate_weights(self, updates, baseline_weights, weights_received):  # pylint: disable=unused-argument
        """Aggregates weights of models with different architectures."""
        self.process_begin = time.time()
        client_id_list = [update.client_id for update in self.updates]
        num_samples = [update.report.num_samples for update in self.updates]
        self.neg_ratio = self.algorithm.nas_aggregation(
            self.subnets_config, weights_received, client_id_list, num_samples
        )
        for payload, client_id in zip(weights_received, client_id_list):
            payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
            self.model_size[client_id - 1] = payload_size

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

        epoch_index = self.algorithm.model.extract_index(subnet_configs)
        self.algorithm.model.step(
            [accuracy_list, round_time_list, self.neg_ratio],
            epoch_index,
            client_id_list,
        )

        self.trainer.model = self.algorithm.model
        self.process_end = time.time()

    def save_to_checkpoint(self) -> None:
        save_config = f"{Config().params['model_path']}/subnet_configs.pickle"
        with open(save_config, "wb") as file:
            pickle.dump(self.subnets_config, file)
        save_config = f"{Config().params['model_path']}/baselines.pickle"
        with open(save_config, "wb") as file:
            pickle.dump(self.algorithm.model.baseline, file)
        return super().save_to_checkpoint()

    def get_logged_items(self) -> dict:
        logged_items = super().get_logged_items()
        acc_info = self.algorithm.get_baseline_accuracy_info()
        logged_items["clients_accuracy_mean"] = acc_info["mean"]
        logged_items["clients_accuracy_std"] = acc_info["std"]
        logged_items["clients_accuracy_max"] = acc_info["max"]
        logged_items["clients_accuracy_min"] = acc_info["min"]
        logged_items["server_overhead"] = self.process_end - self.process_begin
        logged_items["model_size"] = np.mean(self.model_size)
        return logged_items
