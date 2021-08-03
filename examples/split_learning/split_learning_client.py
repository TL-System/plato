"""
A split learning client.
"""

import logging
from dataclasses import dataclass

from plato.clients import simple
from plato.config import Config


@dataclass
class Report:
    """Client report sent to the split learning server."""
    num_samples: int
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

        self.model_received = False
        self.gradient_received = False

    def load_payload(self, server_payload):
        """Loading the server model onto this client."""
        if self.model_received and self.gradient_received:
            self.model_received = False
            self.gradient_received = False

        if not self.model_received:
            self.model_received = True
            self.algorithm.load_weights(server_payload)
        elif not self.gradient_received:
            self.gradient_received = True
            self.algorithm.receive_gradients(server_payload)

    async def train(self):
        """A split learning client only uses the first several layers in a forward pass."""
        logging.info("Training on split learning client #%d", self.client_id)

        assert not Config().clients.do_test

        if self.gradient_received == False:
            # Perform a forward pass till the cut layer in the model
            features = self.algorithm.extract_features(
                self.trainset, self.sampler,
                Config().algorithm.cut_layer)

            # Generate a report for the server, performing model testing if applicable
            return Report(self.sampler.trainset_size(), "features"), features
        else:
            # Perform a complete training with gradients received
            config = Config().trainer._asdict()
            self.algorithm.complete_train(config, self.trainset, self.sampler,
                                          Config().algorithm.cut_layer)
            weights = self.algorithm.extract_weights()
            # Generate a report, signal the end of train
            return Report(self.sampler.trainset_size(), "weights"), weights
