"""
A federated learning client for MistNet.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
import time
from dataclasses import dataclass

from plato.config import Config
from plato.clients import simple


@dataclass
class Report(simple.Report):
    """Client report sent to the MistNet federated learning server."""
    payload_length: int


class Client(simple.Client):
    """A federated learning client for MistNet."""
    async def train(self):
        """A MistNet client only uses the first several layers in a forward pass."""
        logging.info("Training on MistNet client #%d", self.client_id)

        # Since training is performed on the server, the client should not be doing
        # its own testing for the model accuracy
        assert not Config().clients.do_test

        tic = time.perf_counter()

        # Perform a forward pass till the cut layer in the model
        features = self.algorithm.extract_features(
            self.trainset, self.sampler,
            Config().algorithm.cut_layer)

        training_time = time.perf_counter() - tic

        # Generate a report for the server, performing model testing if applicable
        return Report(self.sampler.trainset_size(), 0, training_time, False,
                      len(features)), features
