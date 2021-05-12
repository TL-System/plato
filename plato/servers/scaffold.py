"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
(https://arxiv.org/pdf/1910.06378.pdf)
"""

from plato.config import Config

from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the SCAFFOLD algorithm."""
    def __init__(self):
        super().__init__()
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

    def federated_averaging(self, reports):
        """Aggregate weight updates and deltas updates from the clients."""

        updated_weights = super().federated_averaging(reports)

        # Initialize server update direction
        if self.server_update_direction is None:
            self.server_update_direction = [0] * len(
                self.update_directions_received[0])

        # Update server update direction
        for update_direction in self.update_directions_received:
            for j, delta in enumerate(update_direction):
                self.server_update_direction[j] += delta / Config(
                ).clients.total_clients

        return updated_weights

    async def customize_server_response(self, server_response):
        """Add 'payload_length' into the server response."""
        server_response['payload_length'] = 2

        return server_response

    async def customize_server_payload(self, payload):
        "Add server update direction into the server payload."
        payload_list = []
        payload_list.append(payload)
        payload_list.append(self.server_update_direction)

        return payload_list
