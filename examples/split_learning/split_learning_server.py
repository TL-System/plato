"""
A split learning server.
"""

import logging
import os

import torch
from plato.config import Config
from plato.datasources import feature
from plato.samplers import all_inclusive
from plato.servers import fedavg


class Server(fedavg.Server):
    """The split learning server."""

    def load_gradients(self):
        """ Loading gradients from a file. """
        model_path = Config().params['model_path']
        model_name = Config().trainer.model_name

        model_gradients_path = f'{model_path}/{model_name}_gradients.pth'
        logging.info("[Server #%d] Loading gradients from %s.", os.getpid(),
                     model_gradients_path)

        return torch.load(model_gradients_path)

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
