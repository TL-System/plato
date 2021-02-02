"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import time
from config import Config
from trainers import registry as trainers_registry
from clients import Client
from clients.simple import Report


class EdgeClient(Client):
    """A federated learning client at the edge server in a cross-silo training workload."""
    def __init__(self, server):
        super().__init__()
        self.server = server
        self.trainer = None

    def configure(self):
        """Prepare this edge client for training."""
        self.trainer = trainers_registry.get(self.server.model)

    def load_data(self):
        """The edge client does not need to train models using local data."""

    def load_payload(self, server_payload):
        """The edge client does not need to train models using local data."""

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if 'fedrl' in server_response:
            # Update the number of local aggregation rounds
            Config().algorithm = Config().algorithm._replace(
                local_rounds=server_response['fedrl'])

        if 'current_global_round' in server_response:
            self.server.current_global_round = server_response[
                'current_global_round']
        if 'local_epoch_num' in server_response:
            # Update the number of local epochs
            local_epoch_num = server_response['local_epoch_num'][
                int(self.client_id) - Config().clients.total_clients - 1]
            Config().trainer = Config().trainer._replace(
                epochs=local_epoch_num)

    async def train(self):
        """The aggregation workload on an edge client."""
        training_start_time = time.time()
        # Signal edge server to select clients to start a new round of local aggregation
        self.server.new_global_round_begins.set()

        # Wait for the edge server to finish model aggregation
        await self.server.model_aggregated.wait()
        self.server.model_aggregated.clear()

        # Extract model weights and biases
        weights = self.trainer.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.server.accuracy
        else:
            accuracy = 0

        training_time = time.time() - training_start_time

        return Report(self.client_id, self.server.total_samples, weights,
                      accuracy, training_time, 0)
