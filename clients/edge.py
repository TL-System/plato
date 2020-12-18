"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import asyncio

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

    async def waiter(self, event):
        """
        Wait until the asyncio.Event object is set to be true
        with the set() method.
        """
        await event.wait()

    async def train(self, rl_tuned_para_name=None, rl_tuned_para_value=None):
        """The aggregation workload on an edge client."""
        edge_agg_num = Config().cross_silo.rounds
        if rl_tuned_para_name == 'edge_agg_num':
            edge_agg_num = rl_tuned_para_value
            self.server.edge_agg_num = edge_agg_num

        # self.server.all_local_agg_rounds_done is set to False
        # right before clients/base.py calling self.train()
        while not self.server.all_local_agg_rounds_done:
            await asyncio.sleep(1)

        self.server.current_round = 0

        # Extract model weights and biases
        weights = trainer.extract_weights(self.server.model)

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.server.accuracy
        else:
            accuracy = 0

        return Report(self.client_id, self.server.total_samples, weights,
                      accuracy)
