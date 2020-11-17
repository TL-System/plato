"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import logging
import random
import time

from config import Config
from training import trainer
from clients import Client, Report


class EdgeClient(Client):
    """A federated learning client at the edge server in a cross-silo training workload."""

    def __init__(self, server):
        super().__init__()
        self.server = server


    def configure(self):
        """Prepare this edge client for training."""
        return


    def load_data(self):
        """The edge client does not need to train models using local data."""
        return


    def load_model(self, server_model):
        """Loading the model onto this client."""
        self.server.model.load_state_dict(server_model)


    def train(self):
        """The aggregation workload on an edge client."""
        logging.info('Training on edge client #%s', self.client_id)

        current_round = self.server.current_round

        # Wait for a certain number of aggregation rounds on the edge server
        logging.info("Edge server %s: current round = %s", self.client_id, current_round)
        while current_round == 0 or current_round % Config().cross_silo.rounds != 0:
            time.sleep(1)

        logging.info("Edge server %s: after while", self.client_id)

        # Extract model weights and biases
        weights = trainer.extract_weights(self.server.model)

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.server.accuracy
        else:
            accuracy = 0

        return Report(self.client_id, self.server.total_samples, weights, accuracy)
