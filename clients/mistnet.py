"""
A federated learning client for MistNet.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local Differential Privacy"

"""

import logging
import random

from models import registry as models_registry
from datasets import registry as datasets_registry
from dividers import iid, biased, sharded
from utils import dists
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

    def load_payload(self, server_payload):
        """Loading the server payload as a pre-trained model onto this client."""
        logging.info("Loading a pre-trained model from the server.")
        self.model.load_state_dict(server_payload)

    async def train(self):
        """A MistNet client only uses the first several layers in a forward pass."""
        logging.info('Training on MistNet client #%s', self.client_id)

        # Perform a forward pass till the cut layer in the model
        features = trainer.extract_features(self.model,
                                            Config().training.cut_layer,
                                            self.trainset)

        # Generate a report for the server, performing model testing if applicable
        return Report(self.client_id, len(self.trainset), features)
