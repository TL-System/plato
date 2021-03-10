"""
A federated learning client with support for Adaptive Parameter Freezing (APF).

Reference:

C. Chen, et al. "Communication-Efficient Federated Learning with Adaptive
Parameter Freezing," found in docs/papers.
"""
from dataclasses import dataclass

from config import Config
from clients import SimpleClient


@dataclass
class Report:
    """Client report sent to the federated learning server."""
    num_samples: int
    accuracy: float


class AdaptiveFreezingClient(SimpleClient):
    """A federated learning client with Adaptive Parameter Freezing."""
    async def train(self):
        """Adaptive Parameter Freezing will be applied after training the model."""

        # Perform model training
        self.trainer.train(self.trainset)

        # Extract model weights and biases, with some weights frozen
        weights = self.algorithm.compress_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.trainer.test(self.testset)
        else:
            accuracy = 0

        return Report(len(self.data), accuracy), weights
