"""
A federated learning client for MistNet.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
from dataclasses import dataclass

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
    async def train(self):
        """A MistNet client only uses the first several layers in a forward pass."""
        logging.info("Training on MistNet client #%s", self.client_id)

        # Since training is performed on the server, the client should not be doing
        # its own testing for the model accuracy
        assert not Config().clients.do_test

        # Perform a forward pass till the cut layer in the model
        features = self.trainer.extract_features(
            self.trainset,
            Config().algorithm.cut_layer,
            epsilon=Config().algorithm.epsilon)

        # Generate a report for the server, performing model testing if applicable
        return Report(self.client_id, len(self.trainset), features)
