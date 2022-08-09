"""
A split learning client.
"""

import logging
import time
from dataclasses import dataclass

from plato.clients import simple
from plato.config import Config


@dataclass
class Report(simple.Report):
    """Client report sent to the split learning server."""
    phase: str


class Client(simple.Client):
    """ A split learning client. """

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model=model,
                         datasource=datasource,
                         algorithm=algorithm,
                         trainer=trainer)

        self.phase = 'extract_features'

    def load_payload(self, server_payload):
        """Loading the server model onto this client."""
        payload, info = server_payload

        if info == 'weights':
            self.algorithm.load_weights(payload)
            self.phase = 'extract_features'
        elif info == 'gradients':
            self.algorithm.receive_gradients(payload)
            self.phase = 'complete_train'

    async def train(self):
        """A split learning client only uses the first several layers in a forward pass."""
        assert not Config().clients.do_test
        accuracy = 0
        comm_time = time.time()

        if self.phase == 'extract_features':
            # Perform a forward pass till the cut layer in the model
            logging.info(
                "Performing a forward pass till the cut layer on client #%d",
                self.client_id)

            features, training_time = self.algorithm.extract_features(
                self.trainset, self.sampler,
                Config().algorithm.cut_layer)

            logging.info("Finished extracting features.")
            # Generate a report for the server, performing model testing if applicable
            return Report(self.sampler.trainset_size(), accuracy,
                          training_time, comm_time, False,
                          "features"), features
        else:
            self.model_received = False
            self.gradient_received = False
            # Perform a complete training with gradients received
            config = Config().trainer._asdict()
            training_time = self.algorithm.complete_train(
                config, self.trainset, self.sampler,
                Config().algorithm.cut_layer)
            weights = self.algorithm.extract_weights()
            # Generate a report, signal the end of train
            return Report(self.sampler.trainset_size(), accuracy,
                          training_time, comm_time, False, "weights"), weights
