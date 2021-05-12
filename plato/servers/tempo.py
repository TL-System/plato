"""
A cross-silo federated learning server that tunes
clients' local epoch numbers of each institution.
"""

import math
import statistics

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

        # The central server uses a list to store clients' local epoch numbers of each institution
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

        self.use_min_max(weights_diff_list)
        #self.use_median(weights_diff_list)

    def use_min_max(self, weights_diff_list):
        """A method to compute local epochs."""
        log_list = [-1 * math.log(i) for i in weights_diff_list]
        min_value = min(log_list)
        max_value = max(log_list)

        if min_value == max_value:
            self.local_epoch_list = [
                Config().trainer.epochs
                for i in range(Config().algorithm.total_silos)
            ]
        else:
            max_epoch = 2 * Config().trainer.epochs
            min_epoch = max(int(Config().trainer.epochs / 2), 1)
            a_value = (max_epoch - min_epoch) / (max_value - min_value)
            b_value = min_epoch - a_value * min_value
            self.local_epoch_list = [
                max(1, int(i * a_value + b_value)) for i in log_list
            ]

    def use_median(self, weights_diff_list):
        """A method to compute local epochs."""
        log_list = [-1 * math.log(i) for i in weights_diff_list]
        median = statistics.median(log_list)

        diff = 1

        for i, epoch in enumerate(self.local_epoch_list):
            if epoch > median:
                self.local_epoch_list[i] += diff
            else:
                if self.local_epoch_list[i] > diff:
                    self.local_epoch_list[i] -= diff

    def compute_accuracy_difference(self):
        """
        Compute the absulute value of each edge server's aggregated model accuarcy
        - global model accuracy.
        """
        accuracy_diff_list = []
        for i in range(Config().algorithm.total_silos):
            client_id = i + 1 + Config().clients.total_clients
            accuracy = [
                report.accuracy for (report, __) in self.reports
                if report.client_id == client_id
            ][0]
            accuracy_diff = abs(accuracy - self.accuracy)
            accuracy_diff_list.append(accuracy_diff)
        return accuracy_diff_list

    def get_weights_differences(self):
        """
        Get the weights divergence of each edge server's aggregated model
        and the global model accuracy.
        """
        weights_diff_list = []
        for i in range(Config().algorithm.total_silos):
            client_id = i + 1 + Config().clients.total_clients
            (report, weights) = [(report, payload)
                                 for (report, payload) in self.reports
                                 if int(report.client_id) == client_id][0]
            num_samples = report.num_samples

            weights_diff = self.compute_weights_difference(
                weights, num_samples)

            weights_diff_list.append(weights_diff)

        return weights_diff_list

    def compute_weights_difference(self, local_weights, num_samples):
        """
        Compute the weights divergence of an edge server's aggregated model
        and the global model accuracy.
        """
        weights_diff = 0

        # Extract global model weights
        global_weights = self.algorithm.extract_weights()

        for name, local_weight in local_weights.items():
            global_weight = global_weights[name]
            delta = local_weight - global_weight
            weights_diff += torch.norm(delta).item()

        weights_diff = weights_diff * (num_samples / self.total_samples)

        # global_weights_norm = 0
        # for name, global_weight in global_weights.items():
        #     global_weight = global_weights[name]
        #     global_weights_norm += torch.norm(global_weight).item()

        # weights_diff = weights_diff / global_weights_norm * (
        #     num_samples / self.total_samples)

        return weights_diff
