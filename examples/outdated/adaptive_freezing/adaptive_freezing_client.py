"""
A federated learning client with support for Adaptive Parameter Freezing (APF).

Reference:

C. Chen, H. Xu, W. Wang, B. Li, B. Li, L. Chen, G. Zhang. “Communication-
Efficient Federated Learning with Adaptive Parameter Freezing,” in the
Proceedings of the 41st IEEE International Conference on Distributed Computing
Systems (ICDCS 2021), Online, July 7-10, 2021 (found in papers/).
"""

from plato.config import Config
from plato.clients import simple


class Client(simple.Client):
    """A federated learning client with Adaptive Parameter Freezing."""

    async def _train(self):
        """Adaptive Parameter Freezing will be applied after training the model."""

        # Perform model training
        report, weights = await super().train()

        # Extract model weights and biases, with some weights frozen
        weights = self.algorithm.compress_weights()

        return report, weights
