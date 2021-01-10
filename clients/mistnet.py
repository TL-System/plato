"""
A federated learning client for MistNet.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local Differential Privacy"

"""

import logging
from dataclasses import dataclass

from training import trainer
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
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data),
            set([label for __, label in self.data]))

    async def train(self):
        """A MistNet client only uses the first several layers in a forward pass."""
        logging.info('Training on MistNet client #%s', self.client_id)

        # Perform a forward pass till the cut layer in the model
        features = trainer.extract_features(self.model,
                                            self.trainset,
                                            Config().training.cut_layer,
                                            train=False)

        # Generate a report for the server, performing model testing if applicable
        return Report(self.client_id, len(self.trainset), features)
