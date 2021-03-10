"""
A federated learning server for MistNet.
Reference:
P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
import os
from itertools import chain

from servers import FedAvgServer
from config import Config


class MistNetServer(FedAvgServer):
    """The MistNet server for federated learning."""
    def __init__(self):
        super().__init__()

        # MistNet requires one round of client-server communication
        assert Config().trainer.rounds == 1

    def load_model(self):
        """Setting up a pre-trained model to be loaded on the clients."""
        super().load_model()

        logging.info("[Server #%s] Loading a pre-trained model.", os.getpid())
        self.trainer.load_model()

    async def process_reports(self):
        """Process the features extracted by the client and perform server-side training."""
        features = [features for (__, features) in self.reports]

        # Faster way to deep flatten a list of lists compared to list comprehension
        feature_dataset = list(chain.from_iterable(features))

        # Traing the model using features received from the client
        self.algorithm.train(feature_dataset, Config().algorithm.cut_layer)

        # Test the updated model
        self.accuracy = self.trainer.test(self.testset)
        logging.info('[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
            os.getpid(), 100 * self.accuracy))

        await self.wrap_up_processing_reports()

    @staticmethod
    def is_valid_server_type(server_type):
        """Determine if the server type is valid. """
        return server_type == 'mistnet'

    @staticmethod
    def get_server():
        """Returns an instance of this server. """
        return MistNetServer()
