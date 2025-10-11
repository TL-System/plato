"""
A cross-silo federated learning server that tunes
clients' local epoch numbers of each edge server (institution).
"""

import math

import torch

from plato.config import Config
from plato.servers import fedavg_cs


class Server(fedavg_cs.Server):
    """
    A cross-silo federated learning server that tunes
    clients' local epoch numbers of each edge server.
    """

    def __init__(self):
        super().__init__()

        # The central server uses a list to store each edge server's clients' local epoch numbers
        self.local_epoch_list = None
        if Config().is_central_server():
            self.local_epoch_list = [
                Config().trainer.epochs for i in range(Config().algorithm.total_silos)
            ]

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Wraps up generating the server response with any additional information."""
        server_response = super().customize_server_response(
            server_response, client_id=client_id
        )
        if Config().is_central_server():
            server_response["local_epoch_num"] = self.local_epoch_list
        if Config().is_edge_server():
            # At this point, an edge server already updated Config().trainer.epochs
            # to the number received from the central server.
            # Now it passes the new local epochs number to its clients.
            server_response["local_epoch_num"] = Config().trainer.epochs
        return server_response

    def clients_processed(self):
        """Additional work to be performed after client reports have been processed."""
        super().clients_processed()

        if Config().is_central_server():
            self._update_local_epoch_list()

    def _update_local_epoch_list(self):
        """
        Updates the local epoch list:
        decide clients' local epoch numbers of each edge server.
        """
        weights_diff_list = self.get_weights_differences()

        self._compute_local_epoch(weights_diff_list)

    def _compute_local_epoch(self, weights_diff_list):
        """A method to compute local epochs."""
        # Handle empty weights_diff_list
        if not weights_diff_list:
            self.local_epoch_list = [
                Config().trainer.epochs for i in range(Config().algorithm.total_silos)
            ]
            return

        log_list = [math.log(i) for i in weights_diff_list]
        min_log = min(log_list)
        max_log = max(log_list)

        if min_log == max_log:
            self.local_epoch_list = [
                Config().trainer.epochs for i in range(Config().algorithm.total_silos)
            ]
        else:
            a_value = Config().algorithm.total_silos / 2 / (min_log - max_log)
            b_value = min_log - 4 * max_log
            self.local_epoch_list = [
                max(1, math.ceil(a_value * (3 * i + b_value))) for i in log_list
            ]

    def get_weights_differences(self):
        """
        Gets the weights divergence of each edge server's aggregated model
        and the global model accuracy.
        """
        weights_diff_list = []
        for i in range(Config().algorithm.total_silos):
            client_id = i + 1 + Config().clients.total_clients

            try:
                (report, weights) = [
                    (update.report, update.payload)
                    for update in self.updates
                    if int(update.client_id) == client_id
                ][0]

                num_samples = report.num_samples
                weights_diff = self.compute_weights_difference(weights, num_samples)
                weights_diff_list.append(weights_diff)
            except IndexError:
                # Skip this silo if no updates found
                continue

        return weights_diff_list

    def compute_weights_difference(self, local_weights, num_samples):
        """
        Computes the weight difference of an edge server's aggregated model
        and the global model.
        """
        weights_diff = 0

        # Extract global model weights
        global_weights = self.algorithm.extract_weights()

        for name, local_weight in local_weights.items():
            global_weight = global_weights[name]
            delta = local_weight - global_weight
            delta = delta.float()
            weights_diff += torch.norm(delta).item()

        weights_diff = weights_diff * (num_samples / self.total_samples)

        return weights_diff
