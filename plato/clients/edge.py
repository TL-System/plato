"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import time
from dataclasses import dataclass

from plato.algorithms import registry as algorithms_registry
from plato.config import Config
from plato.trainers import registry as trainers_registry

from plato.clients import base


@dataclass
class Report:
    """Client report, to be sent to the federated learning server."""
    client_id: str
    num_samples: int
    accuracy: float
    training_time: float
    data_loading_time: float


class Client(base.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""
    def __init__(self, server):
        super().__init__()
        self.server = server
        self.trainer = None
        self.algorithm = None

    def configure(self):
        """Prepare this edge client for training."""
        self.trainer = trainers_registry.get()
        self.trainer.set_client_id(self.client_id)

        self.algorithm = algorithms_registry.get(self.trainer)
        self.algorithm.set_client_id(self.client_id)

    def load_data(self):
        """The edge client does not need to train models using local data."""

    def load_payload(self, server_payload):
        """The edge client does not need to train models using local data."""

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if 'current_global_round' in server_response:
            self.server.current_global_round = server_response[
                'current_global_round']

        if 'local_agg_rounds' in server_response:
            # Update the number of local aggregation rounds
            Config().algorithm = Config().algorithm._replace(
                local_rounds=server_response['local_agg_rounds'])

        if 'local_epoch_num' in server_response:
            # Update the number of local epochs
            local_epoch_num = server_response['local_epoch_num'][
                int(self.client_id) - Config().clients.total_clients - 1]
            Config().trainer = Config().trainer._replace(
                epochs=local_epoch_num)

    async def train(self):
        """The aggregation workload on an edge client."""
        training_start_time = time.perf_counter()
        # Signal edge server to select clients to start a new round of local aggregation
        self.server.new_global_round_begins.set()

        # Wait for the edge server to finish model aggregation
        await self.server.model_aggregated.wait()
        self.server.model_aggregated.clear()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.server.accuracy
        else:
            accuracy = 0

        training_time = time.perf_counter() - training_start_time

        return Report(self.client_id, self.server.total_samples, accuracy,
                      training_time, 0), weights
