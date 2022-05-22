"""
A federated learning server for MistNet.
Reference:
P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local
Differential Privacy," found in docs/papers.
"""

import logging
import os

from plato.config import Config
from plato.datasources import feature
from plato.samplers import all_inclusive
from plato.servers import fedavg


class Server(fedavg.Server):
    """The MistNet server for federated learning."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # MistNet requires one round of client-server communication
        assert Config().trainer.rounds == 1

    def load_trainer(self):
        """Setting up a pre-trained model to be loaded on the server."""
        super().load_trainer()

        model_path = Config().params['model_path']
        model_file_name = Config().trainer.pretrained_model if hasattr(
            Config().trainer,
            'pretrained_model') else f'{Config().trainer.model_name}.pth'
        pretrained_model_path = f'{model_path}/{model_file_name}'

        if os.path.exists(pretrained_model_path):
            logging.info("[Server #%d] Loading a pre-trained model.",
                         os.getpid())
            self.trainer.load_model(filename=model_file_name)

    async def process_reports(self):
        """Process the features extracted by the client and perform server-side training."""
        features = [features for (__, __, features, __) in self.updates]
        feature_dataset = feature.DataSource(features)

        # Training the model using all the features received from the client
        sampler = all_inclusive.Sampler(feature_dataset)
        self.algorithm.train(feature_dataset, sampler,
                             Config().algorithm.cut_layer)

        # Test the updated model
        if not hasattr(Config().server, 'do_test') or Config().server.do_test:
            self.accuracy = self.trainer.test(self.testset)
            logging.info('[%s] Global model accuracy: %.2f%%\n', self,
                         100 * self.accuracy)

        await self.wrap_up_processing_reports()
