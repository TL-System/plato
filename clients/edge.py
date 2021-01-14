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

        # The communication time of the central server sending the current model to the edge server
        self.first_communication_time = None

    def configure(self):
        """Prepare this edge client for training."""
        self.trainer = trainers_registry.get(self.server.model)
        return

    def load_data(self):
        """The edge client does not need to train models using local data."""
        return

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if 'fedrl' in server_response:
            # Update the number of local aggregation rounds
            Config().cross_silo = Config().cross_silo._replace(
                rounds=server_response['fedrl'])
            Config().training = Config().training._replace(
                rounds=server_response['fedrl'])
        if 'current_global_round' in server_response:
            self.server.current_global_round = server_response[
                'current_global_round']
        if 'first_communication_start_time' in server_response:
            self.first_communication_time = time.time(
            ) - server_response['first_communication_start_time']

    async def train(self):
        """The aggregation workload on an edge client."""
        self.server.training_time_in_one_global_round = 0
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

        training_time = self.server.training_time_in_one_global_round
        # Send the starting time of second communication (client sending trained model to the server)
        # as the second communication time
        # The server will replace it with the actual time of second communication
        second_communication_start_time = time.time()

        return Report(self.client_id, self.server.total_samples, weights,
                      accuracy, training_time, 0,
                      self.first_communication_time,
                      second_communication_start_time)
