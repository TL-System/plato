"""
A customized server for FedSarah.
"""
from servers import FedAvgServer
from config import Config


class FedSarahServer(FedAvgServer):
    """A federated learning server using the FedSarah algorithm."""
    def __init__(self):
        super().__init__()
        self.server_control_variates = None
        self.control_variates_received = None

    def extract_client_updates(self, reports):
        """Extract the model weights and control variates from clients reports."""

        # Extract the model weights from reports
        weights_received = [payload[0] for (__, payload) in reports]

        # Extract the control variates from reports
        self.control_variates_received = [
            payload[1] for (__, payload) in reports
        ]

        return self.trainer.compute_weight_updates(weights_received)

    def federated_averaging(self, reports):
        """Aggregate weight updates and deltas updates from the clients."""

        updated_weights = super().federated_averaging(reports)

        # Initialize server control variates
        if self.server_control_variates is None:
            self.server_control_variates = [0] * len(
                self.control_variates_received[0])

        # Update server control variates
        for deltas in self.control_variates_received:
            for j, delta in enumerate(deltas):
                self.server_control_variates[j] += delta / Config(
                ).clients.total_clients

        return updated_weights

    async def customize_server_response(self, server_response):
        """Add 'payload_length' into the server response."""
        server_response['payload_length'] = 2

        return server_response

    async def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        payload_list = []
        payload_list.append(payload)
        payload_list.append(self.server_control_variates)

        return payload_list

    @staticmethod
    def is_valid_server_type(server_type):
        """Determine if the server type is valid. """
        return server_type == 'fedsarah'

    @staticmethod
    def get_server():
        """Returns an instance of this server. """
        return FedSarahServer()