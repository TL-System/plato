"""
A basic federated learning client who sends weight updates to the server.
"""

import logging
import time
from dataclasses import dataclass

from algorithms import registry as algorithms_registry
from config import Config
from datasources import registry as datasources_registry
from samplers import registry as samplers_registry
from trainers import registry as trainers_registry

from clients import base


@dataclass
class Report:
    """Client report, to be sent to the federated learning server."""
    num_samples: int
    accuracy: float
    training_time: float
    data_loading_time: float


class Client(base.Client):
    """A basic federated learning client who sends simple weight updates."""
    def __init__(self, model=None, datasource=None, trainer=None):
        super().__init__()
        self.model = model
        self.datasource = datasource
        self.trainer = trainer
        self.trainset = None  # Training dataset
        self.testset = None  # Testing dataset
        self.algorithm = None
        self.sampler = None

        self.data_loading_time = None
        self.data_loading_time_sent = False

    def __repr__(self):
        return 'Client #{}.'.format(self.client_id)

    def configure(self):
        """Prepare this client for training."""
        if self.trainer is None:
            self.trainer = trainers_registry.get(self.client_id, self.model)

        self.algorithm = algorithms_registry.get(self.trainer, self.client_id)

    def load_data(self):
        """Generating data and loading them onto this client."""
        data_loading_start_time = time.time()
        logging.info("[Client #%s] Loading its data source...", self.client_id)

        if self.datasource is None:
            self.datasource = datasources_registry.get()

        self.data_loaded = True

        logging.info("[Client #%s] Dataset size: %s", self.client_id,
                     self.datasource.num_train_examples())

        # Setting up the data sampler
        self.sampler = samplers_registry.get(self.datasource, self.client_id)

        if hasattr(Config().trainer, 'use_mindspore'):
            # MindSpore requires samplers to be used while constructing
            # the dataset
            self.trainset = self.datasource.get_train_set(self.sampler)
        else:
            # PyTorch uses samplers when loading data with a data loader
            self.trainset = self.datasource.get_train_set()

        if Config().clients.do_test:
            # Set the testset if local testing is needed
            self.testset = self.datasource.get_test_set()

        self.data_loading_time = time.time() - data_loading_start_time

    def load_payload(self, server_payload):
        """Loading the server model onto this client."""
        self.algorithm.load_weights(server_payload)

    async def train(self):
        """The machine learning training workload on a client."""
        training_start_time = time.time()
        logging.info("[Client #%s] Started training.", self.client_id)

        # Perform model training
        self.trainer.train(self.trainset, self.sampler)

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.trainer.test(self.testset)
            logging.info("[Client #{:d}] Test accuracy: {:.2f}%".format(
                self.client_id, 100 * accuracy))

        else:
            accuracy = 0

        training_time = time.time() - training_start_time
        data_loading_time = 0

        if not self.data_loading_time_sent:
            data_loading_time = self.data_loading_time
            self.data_loading_time_sent = True

        return Report(self.sampler.trainset_size(), accuracy, training_time,
                      data_loading_time), weights
