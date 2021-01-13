"""
A federated learning client with support for Adaptive Parameter Freezing (APF).

Reference:

C. Chen, et al. "Communication-Efficient Federated Learning with Adaptive Parameter Freezing"

"""

import torch

from models.base import Model
from training import trainer
from config import Config
from clients import SimpleClient
from clients.simple import Report


class APFClient(SimpleClient):
    """A federated learning client with Adaptive Parameter Freezing."""
    def __init__(self):
        super().__init__()
        self.sync_mask = {}

    def load_payload(self, server_payload):
        """Loading the server model onto this client."""
        # Initialize the synchronization mask if necessary
        if not self.sync_mask:
            for name, weight in self.model.to(
                    torch.device('cpu')).named_parameters():
                if weight.requires_grad:
                    self.sync_mask[name] = torch.ones(weight.data.shape).bool()

        # Masking the weights received and load them into the model
        weights_received = []

        for name, weight in server_payload:
            weight.data[self.sync_mask[name]] = weight.data.view(-1)
            weights_received.append((name, weight.data))

        print(weights_received[:2])
        trainer.load_weights(self.model, weights_received)

    def extract_weights(self):
        """Extract weights from a model passed in as a parameter, and apply the APF mask."""
        weights = []
        for name, weight in self.model.to(
                torch.device('cpu')).named_parameters():
            if weight.requires_grad:
                # Rolling back model parameters that should be frozen
                weight.data = torch.where(self.sync_mask == True,
                                          self.weight.data,
                                          self.previous_weight.data)
                # Removing model weights that should not be synced with ther server
                weights_to_sync = torch.masked_select(weight.data,
                                                      self.sync_mask)
                weights.append((name, weights_to_sync))
        return weights

    async def train(self):
        """Adaptive Parameter Freezing will be applied after training the model."""
        # Perform model training
        trainer.train(self.model, self.trainset)

        # Extract model weights and biases
        weights = self.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = trainer.test(self.model, self.testset, 1000)
        else:
            accuracy = 0

        return Report(self.client_id, len(self.data), weights, accuracy)
