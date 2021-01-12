"""
A federated learning client with support for Adaptive Parameter Freezing (APF).

Reference:

C. Chen, et al. "Communication-Efficient Federated Learning with Adaptive Parameter Freezing"

"""

import logging
from dataclasses import dataclass

from training import trainer
from config import Config
from clients import SimpleClient


@dataclass
class Report:
    """Client report sent to the federated learning server."""
    client_id: str
    num_samples: int
    weights: list
    accuracy: float


class APFClient(SimpleClient):
    """A federated learning client with Adaptive Parameter Freezing."""
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data),
            set([label for __, label in self.data]))

    async def train(self):
        """Adaptive Parameter Freezing will be applied after training the model."""
        # Perform model training
        trainer.train(self.model, self.trainset)

        # Extract model weights and biases
        weights = trainer.extract_weights(self.model)

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = trainer.test(self.model, self.testset, 1000)
        else:
            accuracy = 0

        return Report(self.client_id, len(self.data), weights, accuracy)
