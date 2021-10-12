"""
A cross-silo federated learning server that tunes
clients' local epoch numbers of each institution.
"""

import math

import torch
from plato.config import Config

from plato.servers import fedavg_cs


class Server(fedavg_cs.Server):
    """
    A cross-silo federated learning server that tunes
    clients' local epoch numbers of each institution.
    """
    def __init__(self):
        super().__init__()

        # The central server uses a list to store each institution's clients' local epoch numbers
        self.local_epoch_list = None
        if Config().is_central_server():
            self.local_epoch_list = [
                Config().trainer.epochs
                for i in range(Config().algorithm.total_silos)
            ]

        if hasattr(Config(), 'results'):
            if Config().is_edge_server():
                if 'local_epoch_num' not in self.recorded_items:
                    self.recorded_items = self.recorded_items + [
                        'local_epoch_num'
                    ]

    async def customize_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        if Config().is_central_server():
            server_response = await super().customize_server_response(
                server_response)
            server_response['local_epoch_num'] = self.local_epoch_list
        if Config().is_edge_server():
            # At this point, an edge server already updated Config().trainer.epochs
            # to the number received from the central server.
            # Now it passes the new local epochs number to its clients.
            server_response['local_epoch_num'] = Config().trainer.epochs
        return server_response

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        await super().wrap_up_processing_reports()

        if Config().is_central_server():
            self.update_local_epoch_list()

    def update_local_epoch_list(self):
        """
        Update the local epoch list:
        decide clients' local epoch numbers of each institution.
        """
        weights_diff_list = self.get_weights_differences()

        self.compute_local_epoch(weights_diff_list)

    def compute_local_epoch(self, weights_diff_list):
        """A method to compute local epochs."""
        log_list = [math.log(i) for i in weights_diff_list]
        min_log = min(log_list)
        max_log = max(log_list)

        if min_log == max_log:
            self.local_epoch_list = [
                Config().trainer.epochs
                for i in range(Config().algorithm.total_silos)
            ]
        else:
            a_value = Config().algorithm.total_silos / 2 / (min_log - max_log)
            b_value = min_log - 4 * max_log
            self.local_epoch_list = [
                max(1, math.ceil(a_value * (3 * i + b_value)))
                for i in log_list
            ]

    def get_weights_differences(self):
        """
        Get the weights divergence of each edge server's aggregated model
        and the global model accuracy.
        """
        weights_diff_list = []
        for i in range(Config().algorithm.total_silos):
            if hasattr(Config().clients,
                       'simulation') and Config().clients.simulation:
                client_id = i + 1 + Config().clients.per_round
            else:
                client_id = i + 1 + Config().clients.total_clients
            (report, weights) = [(report, payload)
                                 for (report, payload) in self.updates
                                 if int(report.client_id) == client_id][0]
            num_samples = report.num_samples

            weights_diff = self.compute_weights_difference(
                weights, num_samples)

            weights_diff_list.append(weights_diff)

        return weights_diff_list

    def compute_weights_difference(self, local_weights, num_samples):
        """
        Compute the weight difference of an edge server's aggregated model
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
