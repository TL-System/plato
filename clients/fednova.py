"""
A federated learning client whose local number of epochs is randomly
generated and communicated to the server at each communication round.
"""

import logging
import time
import random

from dataclasses import dataclass
from trainers import registry as trainers_registry
from models import registry as models_registry
from config import Config
from clients import simple


@dataclass
class Report(simple.Report):
    """A client report containing the number of local epochs."""
    epochs: list


class FedNovaClient(simple.SimpleClient):
    """A fednova federated learning client who sends weight updates
    and the number of local epochs."""
    def __init__(self):
        super().__init__()
        self.epochs = None
        self.pattern = None
        self.max_local_iter = None
        random.seed(3000 + int(self.client_id))

    def configure(self):
        """Prepare this client for training."""
        model_name = Config().trainer.model
        self.model = models_registry.get(model_name)
        # seperate models with trainers
        self.trainer = trainers_registry.get(self.model)

    async def train(self):
        """The machine learning training workload on a client."""
        training_start_time = time.time()

        # generate the number of local epochs randomly
        if Config().algorithm.pattern == "constant":
            local_epochs = Config().algorithm.max_local_epochs
        else:
            local_epochs = random.randint(2,
                                          Config().algorithm.max_local_epochs)

        logging.info('[Client #%s] Training with %d epoches.', self.epochs,
                     self.client_id)

        # Perform model training for a specific number of epoches
        Config().trainer = Config().trainer._replace(epochs=local_epochs)
        self.trainer.train(self.trainset)

        # Extract model weights and biases
        weights = self.trainer.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.trainer.test(self.testset)
        else:
            accuracy = 0

        training_time = time.time() - training_start_time
        data_loading_time = 0
        if not self.data_loading_time_sent:
            data_loading_time = self.data_loading_time
            self.data_loading_time_sent = True

        return Report(self.client_id, len(self.data), weights, accuracy,
                      training_time, data_loading_time, self.epochs)
