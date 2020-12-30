"""
A basic federated learning client who sends weight updates to the server.
"""

import logging
import random

from models import registry as models_registry
from datasets import registry as datasets_registry
from dividers import iid, biased, sharded
from utils import dists
from training import trainer
from config import Config
from clients import Client, Report


class SimpleClient(Client):
    """A basic federated learning client who sends simple weight updates."""
    def __init__(self):
        super().__init__()
        self.data = None  # The dataset to be used for local training
        self.trainset = None  # Training dataset
        self.testset = None  # Testing dataset

    def __repr__(self):
        return 'Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data),
            set([label for __, label in self.data]))

    def configure(self):
        """Prepare this client for training."""
        model_name = Config().training.model
        self.model = models_registry.get(model_name)

    def load_data(self):
        """Generating data and loading them onto this client."""
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

        # Extract data partition for client
        if Config().data.divider == 'iid':
            assert Config().data.partition_size
            partition_size = Config().data.partition_size
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

        # Extract test parameter settings from the configuration
        test_partition = Config().clients.test_partition

        # Extract the trainset and testset if local testing is needed
        if Config().clients.do_test:
            self.trainset = self.data[:int(
                len(self.data) * (1 - test_partition))]
            self.testset = self.data[
                int(len(self.data) * (1 - test_partition)):]
        else:
            self.trainset = self.data

    def load_model(self, server_model):
        """Loading the model onto this client."""
        self.model.load_state_dict(server_model)

    async def train(self):
        """The machine learning training workload on a client."""
        logging.info('Training on client #%s', self.client_id)

        # Perform model training
        trainer.train(self.model, self.trainset)

        # Extract model weights and biases
        weights = trainer.extract_weights(self.model)

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = trainer.test(self.model, self.testset, 1000)
        else:
            accuracy = 0

        return Report(self.client_id, len(self.data), weights, accuracy)
