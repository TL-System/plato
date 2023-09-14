"""
A federated learning client for MistNet.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
import time
from types import SimpleNamespace

from plato.config import Config
from plato.clients import simple


class Client(simple.Client):
    """A federated learning client for MistNet."""

    async def _train(self):
        """A MistNet client only uses the first several layers in a forward pass."""
        logging.info("Training on MistNet client #%d", self.client_id)

        # Since training is performed on the server, the client should not be doing
        # its own testing for the model accuracy
        assert not Config().clients.do_test

        tic = time.perf_counter()

        # Perform a forward pass till the cut layer in the model
        features = self.algorithm.extract_features(self.trainset, self.sampler)

        training_time = time.perf_counter() - tic

        # Generate a report for the server, performing model testing if applicable
        comm_time = time.time()
        return (
            SimpleNamespace(
                client_id=self.client_id,
                num_samples=self.sampler.num_samples(),
                accuracy=0,
                training_time=training_time,
                comm_time=comm_time,
                update_response=False,
                payload_length=len(features),
            ),
            features,
        )
