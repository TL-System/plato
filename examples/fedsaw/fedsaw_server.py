"""
A cross-silo federated learning server using FedSaw,
as either central or edge servers.
"""

import math
import statistics

import torch

from plato.config import Config
from plato.servers import fedavg_cs


class Server(fedavg_cs.Server):
    """Cross-silo federated learning server using FedSaw."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # The central server uses a list to store each edge server's clients' pruning amount
        self.pruning_amount_list = None

        if Config().is_central_server():
            init_pruning_amount = (
                Config().clients.pruning_amount
                if hasattr(Config().clients, "pruning_amount")
                else 0.4
            )
            self.pruning_amount_list = {
                client_id: init_pruning_amount
                for client_id in range(
                    1 + Config().clients.total_clients,
                    Config().algorithm.total_silos + 1 + Config().clients.total_clients,
                )
            }

        if Config().is_edge_server():
            self.edge_pruning_amount = 0

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Wraps up generating the server response with any additional information."""
        server_response = super().customize_server_response(
            server_response, client_id=client_id
        )
        if Config().is_central_server():
            server_response["pruning_amount"] = self.pruning_amount_list
        if Config().is_edge_server():
            server_response["pruning_amount"] = self.edge_pruning_amount

        return server_response

    # pylint: disable=unused-argument
    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregates the reported weight updates from the selected clients."""
        deltas = await self.aggregate_deltas(updates, weights_received)
        updated_weights = self.algorithm.update_weights(deltas)
        return updated_weights

    def update_pruning_amount_list(self):
        """Updates the list of each institution's clients' pruning_amount."""
        weights_diff_dict, weights_diff_list = self.get_weights_differences()

        median = statistics.median(weights_diff_list)

        for client_id in weights_diff_dict:
            if weights_diff_dict[client_id]:
                self.pruning_amount_list[client_id] = 1 / (
                    1 + math.exp((median - weights_diff_dict[client_id]) / median)
                )

    def get_weights_differences(self):
        """
        Gets the weights differences of each edge server's aggregated model
        and the global model.
        """
        weights_diff_dict = {
            client_id: None
            for client_id in range(
                1 + Config().clients.total_clients,
                Config().algorithm.total_silos + 1 + Config().clients.total_clients,
            )
        }

        weights_diff_list = []

        for update in self.updates:
            client_id = update.report.client_id
            num_samples = update.report.num_samples
            received_updates = update.payload

            weights_diff = self.compute_weights_difference(
                received_updates, num_samples
            )

            weights_diff_dict[client_id] = weights_diff
            weights_diff_list.append(weights_diff)

        return weights_diff_dict, weights_diff_list

    def compute_weights_difference(self, received_updates, num_samples):
        """
        Computes the weight difference of an edge server's aggregated model
        and the global model.
        """
        weights_diff = 0

        for _, delta in received_updates.items():
            delta = delta.float()
            weights_diff += torch.norm(delta).item()

        weights_diff = weights_diff * (num_samples / self.total_samples)

        return weights_diff

    def get_logged_items(self) -> dict:
        """Gets items to be logged by the LogProgressCallback class in a .csv file."""
        logged_items = super().get_logged_items()
        logged_items["pruning_amount"] = (
            self.edge_pruning_amount if Config().is_edge_server() else 0
        )

        return logged_items

    def clients_processed(self):
        """Additional work to be performed after client reports have been processed."""
        super().clients_processed()

        if Config().is_central_server():
            self.update_pruning_amount_list()
