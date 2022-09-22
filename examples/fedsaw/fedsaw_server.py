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

        # The central server uses a list to store each institution's clients' pruning amount
        self.pruning_amount_list = None

        if Config().is_central_server():
            self.init_pruning_amount = (
                Config().clients.pruning_amount
                if hasattr(Config().clients, "pruning_amount")
                else 0.4
            )
            self.pruning_amount_list = [
                self.init_pruning_amount for i in range(Config().algorithm.total_silos)
            ]

        if Config().is_edge_server():
            self.edge_pruning_amount = 0
            if (
                hasattr(Config(), "results")
                and "pruning_amount" not in self.recorded_items
            ):
                self.recorded_items = self.recorded_items + ["pruning_amount"]

    def customize_server_response(self, server_response: dict) -> dict:
        """Wrap up generating the server response with any additional information."""
        if Config().is_central_server():
            server_response["pruning_amount"] = self.pruning_amount_list
        if Config().is_edge_server():
            server_response["pruning_amount"] = self.edge_pruning_amount

        return server_response

    # pylint: disable=unused-argument
    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate the reported weight updates from the selected clients."""
        deltas = await self.aggregate_deltas(updates, weights_received)
        updated_weights = self.algorithm.update_weights(deltas)
        return updated_weights

    def update_pruning_amount_list(self):
        """Update the list of each institution's clients' pruning_amount."""
        weights_diff_list = self.get_weights_differences()

        median = statistics.median(weights_diff_list)

        for i, weight_diff in enumerate(weights_diff_list):
            if weight_diff >= median:
                self.pruning_amount_list[i] = self.init_pruning_amount * (
                    1 + math.tanh(weight_diff / sum(weights_diff_list))
                )
            else:
                self.pruning_amount_list[i] = self.init_pruning_amount * (
                    1 - math.tanh(weight_diff / sum(weights_diff_list))
                )

    def get_weights_differences(self):
        """
        Get the weights differences of each edge server's aggregated model
        and the global model.
        """
        weights_diff_list = []
        for i in range(Config().algorithm.total_silos):
            client_id = i + 1 + Config().clients.total_clients

            (report, received_updates) = [
                (update.report, update.payload)
                for update in self.updates
                if int(update.report.client_id) == client_id
            ][0]
            num_samples = report.num_samples

            weights_diff = self.compute_weights_difference(
                received_updates, num_samples
            )

            weights_diff_list.append(weights_diff)

        return weights_diff_list

    def compute_weights_difference(self, received_updates, num_samples):
        """
        Compute the weight difference of an edge server's aggregated model
        and the global model.
        """
        weights_diff = 0

        for _, delta in received_updates.items():
            delta = delta.float()
            weights_diff += torch.norm(delta).item()

        weights_diff = weights_diff * (num_samples / self.total_samples)

        return weights_diff

    def get_record_items_values(self):
        """Get values that will be recorded in result csv file."""
        record_items_values = super().get_record_items_values()
        record_items_values["pruning_amount"] = (
            self.edge_pruning_amount if Config().is_edge_server() else 0
        )

        return record_items_values

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        await super().wrap_up_processing_reports()

        if Config().is_central_server():
            self.update_pruning_amount_list()
