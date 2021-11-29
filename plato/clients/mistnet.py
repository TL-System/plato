"""
A federated learning client for MistNet.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
from dataclasses import dataclass

from plato.config import Config
from plato.clients import simple
from plato.dataprocessor import registry as dataprocessor_registry


@dataclass
class Report:
    """Client report sent to the MistNet federated learning server."""
    num_samples: int
    payload_length: int


class Client(simple.Client):
    """A federated learning client for MistNet."""
    async def train(self):
        """A MistNet client only uses the first several layers in a forward pass."""
        logging.info("Training on MistNet client #%d", self.client_id)

        # Since training is performed on the server, the client should not be doing
        # its own testing for the model accuracy
        assert not Config().clients.do_test

        # Perform a forward pass till the cut layer in the model
        features = self.algorithm.extract_features(
            self.trainset,
            self.sampler,
            Config().algorithm.cut_layer,
            epsilon=Config().algorithm.epsilon)

        # Generate a report for the server, performing model testing if applicable
        return Report(self.sampler.trainset_size(), len(features)), features

    def set_dataprocessors(self) -> None:
        """Prepare this client for training."""
        self.send_dataprocessor, self.receive_dataprocessor = dataprocessor_registry.get(
            "client",
            trainer=self.trainer,
            epsilon=Config().algorithm.epsilon,
            client_id=self.client_id)
