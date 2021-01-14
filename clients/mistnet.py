"""
A federated learning client for MistNet.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
from dataclasses import dataclass

from models import registry as models_registry
from trainers import registry as trainers_registry
from config import Config
from clients import SimpleClient


@dataclass
class Report:
    """Client report sent to the MistNet federated learning server."""
    client_id: str
    num_samples: int
    features: list


class MistNetClient(SimpleClient):
    """A federated learning client for MistNet."""
    def configure(self):
        """Prepare this client for training."""
        model_name = Config().training.model
        self.model = models_registry.get(model_name)
        self.trainer = trainers_registry.get(self.model)

    async def train(self):
        """A MistNet client only uses the first several layers in a forward pass."""
        logging.info('Training on MistNet client #%s', self.client_id)

        # Perform a forward pass till the cut layer in the model
        features = self.trainer.extract_features(
            self.trainset,
            Config().training.cut_layer,
            epsilon=Config().training.epsilon)

        # Generate a report for the server, performing model testing if applicable
        return Report(self.client_id, len(self.trainset), features)
