"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," 
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""
import logging
import os

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.server_update_direction = None
        self.update_directions_received = None

    def extract_client_updates(self, reports):
        """Extract the model weights and update directions from clients reports."""

        # Extract the model weights from reports
        weights_received = [payload[0] for (__, payload) in reports]

        # Extract the update directions from reports
        self.update_directions_received = [
            payload[1] for (__, payload) in reports
        ]

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        """Aggregate weight updates and deltas updates from the clients."""

        update = await super().federated_averaging(updates)

        # Initialize server update direction
        if self.server_update_direction is None:
            self.server_update_direction = [0] * len(
                self.update_directions_received[0])

        # Update server update direction
        for update_direction in self.update_directions_received:
            for j, delta in enumerate(update_direction):
                self.server_update_direction[j] += delta / Config(
                ).clients.total_clients

        return update

    def customize_server_payload(self, payload):
        """ Add server update direction into the server payload. """
        custom_payload = super().customize_server_payload(payload)
        if isinstance(custom_payload, list):
            custom_payload.append(self.server_update_direction)
        elif isinstance(custom_payload, dict):
            custom_payload[
                'server_update_direction'] = self.server_update_direction
        else:
            logging.info(
                "Server payload is neither a list nor a dictionary. Aborting.")
            os.exit()

        return custom_payload
