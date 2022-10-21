"""
A federated learning client using split learning.

Reference:

Vepakomma, et al., "Split learning for health: Distributed deep learning without sharing
raw patient data," in Proc. AI for Social Good Workshop, affiliated with ICLR 2018.

https://arxiv.org/pdf/1812.00564.pdf
"""

import logging
import time
from types import SimpleNamespace

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A split learning client."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        assert not Config().clients.do_test

        self.model_received = False
        self.gradient_received = False

    async def inbound_processed(self, data):
        """Extract features or complete the training using split learning."""
        server_payload, info = data

        # Preparing the client response
        report, payload = None, None

        if info == "weights":
            # server sends the global model, i.e., feature extraction
            report, payload = self._extract_features(server_payload)
        elif info == "gradients":
            # server sends the gradients of the features, i.e., complete training
            report, payload = self._complete_training(server_payload)

        return report, payload

    def _extract_features(self, payload):
        """Extract the feature till the cut layer."""
        self.algorithm.load_weights(payload)
        # Perform a forward pass till the cut layer in the model
        logging.info(
            "Performing a forward pass till the cut layer on client #%d",
            self.client_id,
        )

        features, training_time = self.algorithm.extract_features(
            self.trainset, self.sampler
        )
        logging.info("[%s] Finished extracting features.", self)
        # Generate a report for the server, performing model testing if applicable
        report = SimpleNamespace(
            num_samples=self.sampler.num_samples(),
            accuracy=0,
            training_time=training_time,
            comm_time=time.time(),
            update_response=False,
            phase="features",
        )
        return report, features
    
    def _complete_training(self, payload):
        """Complete the training based on the gradients from server."""
        self.algorithm.receive_gradients(payload)
         # Perform a complete training with gradients received
        config = Config().trainer._asdict()
        training_time = self.algorithm.complete_train(
            config, self.trainset, self.sampler
        )
        weights = self.algorithm.extract_weights()
        # Generate a report, signal the end of train
        report = SimpleNamespace(
            num_samples=self.sampler.num_samples(),
            accuracy=0,
            training_time=training_time,
            comm_time=time.time(),
            update_response=False,
            phase="weights",
        )
        return report, weights
        