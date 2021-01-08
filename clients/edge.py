"""
A federated learning client at the edge server in a cross-silo training workload.
"""

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

    def wrap_up_before_training(self, data):
        """ Wrap up before training in case server response has any additional information."""
        if 'fedrl' in data:
            # Update the number of local aggregation rounds
            Config().cross_silo = Config().cross_silo._replace(
                rounds=data['fedrl'])
            Config().training = Config().training._replace(
                rounds=data['fedrl'])

    async def train(self):
        """The aggregation workload on an edge client."""
        # Signal edge server to select clients to start a new round of local aggregation
        self.server.new_global_round_begins.set()

        # Wait for the edge server to finish model aggregation
        await self.server.model_aggregated.wait()
        self.server.model_aggregated.clear()

        # Extract model weights and biases
        weights = trainer.extract_weights(self.server.model)

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.server.accuracy
        else:
            accuracy = 0

        return Report(self.client_id, self.server.total_samples, weights,
                      accuracy)
