"""
A basic federated learning client who sends weight updates to the server.
"""

import logging
import json
import random
import pickle
import websockets

from models import registry as models_registry
from datasets import registry as datasets_registry
from training import optimizers, trainer
from dividers import iid, biased, sharded
from utils import dists


class SimpleClient:
    """A basic federated learning client who sends simple weight updates."""

    def __init__(self, config, client_id):
        self.config = config
        self.client_id = client_id
        self.do_test = None # Should the client test the trained model?
        self.test_partition = None # Percentage of the dataset reserved for testing
        self.data = None # The dataset to be used for local training
        self.trainset = None # Training dataset
        self.testset = None # Testing dataset
        self.report = None # Report to the server
        self.task = None # Local computation task: 'train' or 'test'
        self.model = None # Machine learning model
        self.epochs = None # The number of epochs in each local training round
        self.batch_size = None # The batch size used for local training
        self.pref = None # Preferred label on this client in biased data distribution
        self.bias = None # Percentage of bias
        self.optimizer = None # Optimizer for model training
        self.loader = None


    def __repr__(self):
        return 'Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data), set([label for __, label in self.data]))


    async def start_client(self):
        """Startup function for a client."""
        uri = 'ws://{}:{}'.format(self.config.server.address, self.config.server.port)

        async with websockets.connect(uri, max_size=2 ** 30) as websocket:
            logging.info("Signing in at the server with client ID %s...", self.client_id)
            await websocket.send(json.dumps({'id': self.client_id}))

            while True:
                logging.info("Waiting to be selected by the server for training...")
                server_response = await websocket.recv()
                data = json.loads(server_response)

                if data['id'] == self.client_id and 'payload' in data:
                    logging.info("Selected by the server for training -- receiving the model...")
                    server_model = await websocket.recv()
                    self.model.load_state_dict(pickle.loads(server_model))

                    self.train()

                    logging.info("Model trained on client with client ID %s...", self.client_id)
                    # Sending client ID as metadata to the server (payload to follow)
                    client_update = {'id': self.client_id, 'payload': True}
                    await websocket.send(json.dumps(client_update))

                    # Sending the client training report to the server as payload
                    await websocket.send(pickle.dumps(self.report))


    def configure(self):
        """Prepare this client for training."""
        self.task = self.config.training.task
        self.epochs = self.config.training.epochs
        self.batch_size = self.config.training.batch_size

        model_name = self.config.training.model
        self.model = models_registry.get(model_name, self.config)
        self.optimizer = optimizers.get_optimizer(self.config, self.model)

        self.load_data()


    def load_data(self):
        """Generating data and loading them onto this client."""
        # Extract configurations for the datasets
        config = self.config

        # Set up the training and testing datasets
        data_path = config.training.data_path
        dataset = datasets_registry.get(config.training.dataset, data_path)

        logging.info('Dataset size: %s', dataset.num_train_examples())
        logging.info('Number of classes: %s', dataset.num_classes())

        # Setting up the data loader
        self.loader = {
            'iid': iid.IIDDivider,
            'bias': biased.BiasedDivider,
            'shard': sharded.ShardedDivider
        }[config.loader](config, dataset)

        logging.info('Data distribution: %s', config.loader)

        is_iid = self.config.data.iid
        labels = self.loader.labels
        loader = self.config.loader
        num_clients = self.config.clients.total

        if not is_iid:  # Create a non-IID distribution for label preferences
            dist, __ = {
                "uniform": dists.uniform,
                "normal": dists.normal
            }[self.config.clients.label_distribution](num_clients, len(labels))
            random.shuffle(dist)  # Shuffle the distribution

        logging.info('Initializing the client data...')

        if not is_iid: # Configure this client for non-IID data
            if self.config.data.bias:
                # Bias data partitions
                self.bias = self.config.data.bias
                # Choose weighted random preference
                self.pref = random.choices(labels, dist)[0]

        logging.info('Total number of clients: %s', num_clients)

        if loader == 'shard': # Create data shards
            self.loader.create_shards()

        loader = self.config.loader

        # Get data partition size
        if loader != 'shard':
            if self.config.data.partition_size:
                partition_size = self.config.data.partition_size

        # Extract data partition for client
        if loader == 'iid':
            self.data = self.loader.get_partition(partition_size)
        elif loader == 'bias':
            self.data = self.loader.get_partition(partition_size, self.pref)
        elif loader == 'shard':
            self.data = self.loader.get_partition()
        else:
            logging.critical('Unknown data loader type.')

        # Extract test parameter settings from the configuration
        do_test = self.do_test = self.config.clients.do_test
        test_partition = self.test_partition = self.config.clients.test_partition

        # Extract the trainset and testset if local testing is needed
        if do_test:
            self.trainset = self.data[:int(len(self.data) * (1 - test_partition))]
            self.testset = self.data[int(len(self.data) * (1 - test_partition)):]
        else:
            self.trainset = self.data


    def train(self):
        """The machine learning training workload on a client."""
        logging.info('Training on client #%s', self.client_id)

        # Perform model training
        trainloader = trainer.get_trainloader(self.trainset, self.batch_size)
        trainer.train(self.model, trainloader,
                       self.optimizer, self.epochs)

        # Extract model weights and biases
        weights = trainer.extract_weights(self.model)

        # Generate a report for the server
        self.report = Report(self, weights)

        # Perform model testing if applicable
        if self.do_test:
            self.test()


    def test(self):
        """Perform model testing."""
        testloader = trainer.get_testloader(self.testset, 1000)
        self.report.set_accuracy(trainer.test(self.model, testloader))
        logging.info("Accuracy: %s", self.report.accuracy)
        return self.report


class Report:
    """Federated learning client report."""

    def __init__(self, client, weights):
        self.client_id = client.client_id
        self.num_samples = len(client.data)
        self.weights = weights
        self.accuracy = 0


    def set_accuracy(self, accuracy):
        """Include the test accuracy computed at a client in the report."""
        self.accuracy = accuracy
