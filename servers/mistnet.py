"""
A federated learning server for MistNet.
Reference:
P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
import os
from itertools import chain
from torch.utils.data import SubsetRandomSampler

from servers import fedavg
from config import Config


class Server(fedavg.Server):
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

        # Training the model using features received from the client
        sampler = AllDataSampler(feature_dataset)
        self.algorithm.train(feature_dataset, sampler,
                             Config().algorithm.cut_layer)

        # Test the updated model
        self.accuracy = self.trainer.test(self.testset)
        logging.info('[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
            os.getpid(), 100 * self.accuracy))

        await self.wrap_up_processing_reports()


class AllDataSampler:
    def __init__(self, dataset):
        self.all_data = range(len(dataset))

    def get(self):
        return SubsetRandomSampler(self.all_data)