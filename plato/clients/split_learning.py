"""
A split learning client.
"""

import logging
from dataclasses import dataclass

from plato.config import Config

from plato.clients import simple


@dataclass
class Report:
    """Client report sent to the split learning server."""
    num_samples: int
    payload_length: int


class Client(simple.Client):
    def __init__(self, model=None, datasource=None, trainer=None):
        super().__init__()

        self.model_received = False
        self.gradient_received = False

    def load_payload(self, server_payload):
        """Loading the server model onto this client."""
        if self.model_received == True and self.gradient_received == True:
            self.model_received = False
            self.gradient_received = False

        if self.model_received == False:
            self.model_received = True
            self.algorithm.load_weights(server_payload)
        elif self.gradient_received == False:
            self.gradient_received = True
            self.algorithm.load_gradients(server_payload)

    async def train(self):
        """A split learning client only uses the first several layers in a forward pass."""
        logging.info("Training on split learning client #%s", self.client_id)

        # Since training is performed on the server, the client should not be doing
        # its own testing for the model accuracy
        assert not Config().clients.do_test

        if self.gradient_received == False:
            # Perform a forward pass till the cut layer in the model
            features = self.algorithm.extract_features(
                self.trainset, self.sampler,
                Config().algorithm.cut_layer)

            # Generate a report for the server, performing model testing if applicable
            return Report(self.sampler.trainset_size(),
                          len(features)), features
        else:
            # Perform a complete training with gradients received
            config = Config().trainer._asdict()
            self.algorithm.complete_train(config, self.trainset, self.sampler,
                                          Config().algorithm.cut_layer)

            # Generate a report, signal the end of train
            train_status = "train done"
            return Report(0, 0), train_status