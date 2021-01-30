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
    def load_model(self):
        """Setting up a pre-trained model to be loaded on the clients."""
        super().load_model()

        logging.info("[Server #%s] Loading a pre-trained model.", os.getpid())
        self.trainer.load_model(Config().trainer.model + ".pth")

    async def process_reports(self):
        """Process the features extracted by the client and perform server-side training."""
        features = [report.features for report in self.reports]

        # Faster way to deep flatten a list of lists compared to list comprehension
        feature_dataset = list(chain.from_iterable(features))

        # Traing the model using features received from the client
        self.trainer.train(feature_dataset, Config().algorithm.cut_layer)

        # Test the updated model
        self.accuracy = self.trainer.test(feature_dataset,
                                          Config().algorithm.cut_layer)
        logging.info("Global model accuracy: {:.2f}%\n".format(100 *
                                                               self.accuracy))

        await self.wrap_up_processing_reports()
