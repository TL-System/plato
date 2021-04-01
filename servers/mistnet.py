"""
A federated learning server for MistNet.
Reference:
P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
import os
from itertools import chain

from servers import fedavg
from config import Config
from torch.utils.data import SubsetRandomSampler


class Server(fedavg.Server):
    """The MistNet server for federated learning."""
    def __init__(self):
        super().__init__()

        # MistNet requires one round of client-server communication
        assert Config().trainer.rounds == 1

    async def process_reports(self):
        """Process the features extracted by the client and perform server-side training."""
        features = [features for (__, features) in self.reports]

        # Faster way to deep flatten a list of lists compared to list comprehension
        feature_dataset = list(chain.from_iterable(features))

        # Training the model using features received from the client
        sampler = Mistnet_Sampler(feature_dataset)
        self.algorithm.train(feature_dataset, sampler, Config().algorithm.cut_layer)

        # Test the updated model
        self.accuracy = self.trainer.test(self.testset)
        logging.info('[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
            os.getpid(), 100 * self.accuracy))

        await self.wrap_up_processing_reports()

class Mistnet_Sampler:
    def __init__(self, dataset):
        self.alldata = range(len(dataset))

    def get(self):
        return SubsetRandomSampler(self.alldata)