"""
A federated learning client whose local iteration is randomly generated and communicated to the server at each communication round.
Notably, MistNet is applied here.
"""

import logging
import time
import numpy as np
from dataclasses import dataclass
from models import registry as models_registry
from config import Config
from clients import simple
from trainers import fednova


@dataclass
class ReportFednova(simple.Report):
    """Client report containing the local iteration sent to the federated learning server."""
    iteration: list


class FedNovaClient(simple.SimpleClient):
    """A fednova federated learning client who sends weight updates and local iterations."""
    def __init__(self):
        super().__init__()
        self.iteration = None
        self.pattern = None
        self.max_local_iter = None

    def configure(self):
        """Prepare this client for training."""
        model_name = Config().trainer.model
        self.model = models_registry.get(model_name)
        # seperate models with trainers
        self.trainer = fednova.Trainer(
            self.model)  #trainers_registry.get(self.model)
        self.pattern = Config().clients.pattern
        self.max_local_iter = Config().clients.max_local_iter

    async def train(self):
        """The machine learning training workload on a client."""
        training_start_time = time.time()

        # generate local iteration randomly
        self.iteration = self.update_local_iteration(self.pattern)
        logging.info('[Client #%s] Training %d epoches.', self.iteration,
                     self.client_id)

        # Perform model training for specific epoches
        self.trainer.train(self.trainset, iteration=self.iteration)

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

        return ReportFednova(self.client_id, len(self.data), weights, accuracy,
                             training_time, data_loading_time, self.iteration)

    def update_local_iteration(self, pattern):
        """ update local epoch for each client"""
        if pattern == "constant":
            return self.max_local_iter

        if pattern == "uniform_random":
            np.random.seed(2020 + int(self.client_id))
            return np.random.randint(low=2, high=self.max_local_iter,
                                     size=1)[0]
