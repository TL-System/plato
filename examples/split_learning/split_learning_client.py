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
from plato.utils import fonts


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
        self.contexts = {}
        self.original_weights = None

        # Iteration control
        self.iterations = Config().clients.iteration
        self.iter_left = Config().clients.iteration

    async def inbound_processed(self, processed_inbound_payload):
        """Extract features or complete the training using split learning."""
        server_payload, info = processed_inbound_payload

        # Preparing the client response
        report, payload = None, None

        if info == "prompt":
            # Server prompts a new client to conduct split learning
            self._load_context(self.client_id)
            self.algorithm.load_data(self.trainset, self.sampler)
            report, payload = self._extract_features()
        elif info == "gradients":
            # server sends the gradients of the features, i.e., complete training
            logging.warning("[%s] Gradients received, complete training.", self)
            training_time, weights = self._complete_training(server_payload)
            self.iter_left -= 1

            if self.iter_left == 0:
                logging.warning("[%s] Finished training, sending weights to the server.", self)
                # Save the state of current client
                self._save_context(self.client_id)
                # Send weights to server for evaluation
                report = SimpleNamespace(
                    num_samples=self.sampler.num_samples(),
                    accuracy=0,
                    training_time=training_time,
                    comm_time=time.time(),
                    update_response=False,
                    type="weights",
                )
                payload = weights
                self.iter_left = self.iterations
            else:
                # Continue feature extraction
                report, payload = self._extract_features()
                report.training_time += training_time
        return report, payload

    def _save_context(self, client_id):
        """Saving the extracted weights for a given client."""
        self.contexts[client_id] = self.algorithm.extract_weights()

    def _load_context(self, client_id):
        """Load client's model weights."""
        if not client_id in self.contexts:
            if self.original_weights is None:
                self.original_weights = self.algorithm.extract_weights()
            self.algorithm.load_weights(self.original_weights)
        else:
            self.algorithm.load_weights(self.contexts.pop(client_id))

    def _extract_features(self):
        """Extract the feature till the cut layer."""
        round_number = self.iterations - self.iter_left + 1
        logging.warning(
            fonts.colourize(
                f"[{self}] Started split learning in round #{round_number}/{self.iterations}"
                + f" (Global round {self.current_round})."
            )
        )

        features, training_time = self.algorithm.extract_features()
        report = SimpleNamespace(
            num_samples=self.sampler.num_samples(),
            accuracy=0,
            training_time=training_time,
            comm_time=time.time(),
            update_response=False,
            type="features",
        )
        return report, features

    def _complete_training(self, payload):
        """Complete the training based on the gradients from server."""
        self.algorithm.receive_gradients(payload)
        # Perform a complete training with gradients received
        training_time = self.algorithm.complete_train()
        weights = self.algorithm.extract_weights()
        return training_time, weights
