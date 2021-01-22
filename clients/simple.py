"""
A basic federated learning client who sends weight updates to the server.
"""

import logging
import random
import os
import time
from dataclasses import dataclass

from models import registry as models_registry
from datasets import registry as datasets_registry
from trainers import registry as trainers_registry
from dividers import iid, biased, sharded
from utils import dists
from config import Config
from clients import Client


@dataclass
class Report:
    """Client report sent to the federated learning server."""
    client_id: str
    num_samples: int
    weights: list
    accuracy: float
    training_time: float
    data_loading_time: float


class SimpleClient(Client):
    """A basic federated learning client who sends simple weight updates."""
    def __init__(self):
        super().__init__()
        self.data = None  # The dataset to be used for local training
        self.trainset = None  # Training dataset
        self.testset = None  # Testing dataset
        self.trainer = None

        self.data_loading_time = None
        self.data_loading_time_sent = False

    def __repr__(self):
        return 'Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data),
            set([label for __, label in self.data]))

    def configure(self):
        """Prepare this client for training."""
        model_type = Config().trainer.model
        self.model = models_registry.get(model_type)
        self.trainer = trainers_registry.get(self.model, self.client_id)

    def load_data(self):
        """Generating data and loading them onto this client."""
        data_loading_start_time = time.time()
        logging.info('Client #%s is loading its dataset...', self.client_id)

        dataset = datasets_registry.get()
        self.data_loaded = True

        logging.info('Dataset size: %s', dataset.num_train_examples())
        logging.info('Number of classes: %s', dataset.num_classes())

        # Setting up the data divider
        assert Config().data.divider in ('iid', 'bias', 'shard')
        logging.info('Data distribution: %s', Config().data.divider)

        divider = {
            'iid': iid.IIDDivider,
            'bias': biased.BiasedDivider,
            'shard': sharded.ShardedDivider
        }[Config().data.divider](dataset)

        num_clients = Config().clients.total_clients

        logging.info("[Client #%s] Extracting the local dataset...",
                     self.client_id)

        # Extract data partition for client
        if Config().data.divider == 'iid':
            assert Config().data.partition_size is not None
            partition_size = Config().data.partition_size

            assert partition_size * Config(
            ).clients.per_round <= dataset.num_train_examples()

            self.data = divider.get_partition(partition_size)

        elif Config().data.divider == 'bias':
            assert Config().data.label_distribution in ('uniform', 'normal')

            dist, __ = {
                "uniform": dists.uniform,
                "normal": dists.normal
            }[Config().data.label_distribution](num_clients,
                                                len(divider.labels))
            random.shuffle(dist)

            pref = random.choices(divider.labels, dist)[0]

            assert Config().data.partition_size
            partition_size = Config().data.partition_size
            self.data = divider.get_partition(partition_size, pref)

        elif Config().data.divider == 'shard':
            self.data = divider.get_partition()

        # Extract the trainset and testset if local testing is needed
        if Config().clients.do_test:
            self.trainset = self.data
            self.testset = divider.testset
        else:
            self.trainset = self.data

        self.data_loading_time = time.time() - data_loading_start_time

    def load_payload(self, server_payload):
        """Loading the server model onto this client."""
        self.trainer.load_weights(server_payload)

    async def train(self):
        """The machine learning training workload on a client."""
        training_start_time = time.time()
        logging.info('[Client %s] Started to train on client #%s', os.getpid(),
                     self.client_id)

        # Perform model training
        self.trainer.train(self.trainset)

        # Extract model weights and biases
        weights = self.trainer.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.trainer.test(self.testset, 1000)
        else:
            accuracy = 0

        training_time = time.time() - training_start_time
        data_loading_time = 0
        if not self.data_loading_time_sent:
            data_loading_time = self.data_loading_time
            self.data_loading_time_sent = True

        return Report(self.client_id, len(self.data), weights, accuracy,
                      training_time, data_loading_time)
