"""
A basic federated learning client who sends weight updates to the server.
"""

import logging
import json
import random
import pickle
from dataclasses import dataclass
import websockets

from models import registry as models_registry
from datasets import registry as datasets_registry
from dividers import iid, biased, sharded
from utils import dists
from training import trainer


@dataclass
class Report:
    """Client report sent to the federated learning server."""
    client_id: str
    num_samples: int
    weights: list
    accuracy: float


class SimpleClient:
    """A basic federated learning client who sends simple weight updates."""

    def __init__(self, config, client_id):
        self.config = config
        self.client_id = client_id
        self.data = None # The dataset to be used for local training
        self.trainset = None # Training dataset
        self.testset = None # Testing dataset
        self.model = None # Machine learning model
        self.data_loaded = False # is training data already loaded from the disk?


    def __repr__(self):
        return 'Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data), set([label for __, label in self.data]))


    async def start_client(self):
        """Startup function for a client."""
        uri = 'ws://{}:{}'.format(self.config.server.address, self.config.server.port)
        try:
            async with websockets.connect(uri, max_size=2 ** 30) as websocket:
                logging.info("Signing in at the server with client ID %s...", self.client_id)
                await websocket.send(json.dumps({'id': self.client_id}))

                while True:
                    logging.info("Client %s is waiting to be selected...", self.client_id)
                    server_response = await websocket.recv()
                    data = json.loads(server_response)

                    if data['id'] == self.client_id and 'payload' in data:
                        logging.info("Client %s has been selected and receiving the model...",
                                    self.client_id)
                        server_model = await websocket.recv()
                        self.model.load_state_dict(pickle.loads(server_model))

                        if not self.data_loaded:
                            self.load_data()

                        report = self.train()

                        logging.info("Model trained on client with client ID %s.", self.client_id)
                        # Sending client ID as metadata to the server (payload to follow)
                        client_update = {'id': self.client_id, 'payload': True}
                        await websocket.send(json.dumps(client_update))

                        # Sending the client training report to the server as payload
                        await websocket.send(pickle.dumps(report))
        except OSError:
            logging.info("Client #%s: connection to the server failed.",
                self.client_id)


    def configure(self):
        """Prepare this client for training."""
        model_name = self.config.training.model
        self.model = models_registry.get(model_name, self.config)


    def load_data(self):
        """Generating data and loading them onto this client."""
        config = self.config
        data_path = config.training.data_path
        dataset = datasets_registry.get(config.training.dataset, data_path)
        self.data_loaded = True

        logging.info('Dataset size: %s', dataset.num_train_examples())
        logging.info('Number of classes: %s', dataset.num_classes())

        # Setting up the data loader
        loader = {
            'iid': iid.IIDDivider,
            'bias': biased.BiasedDivider,
            'shard': sharded.ShardedDivider
        }[config.loader](config, dataset)

        logging.info('Data distribution: %s', config.loader)

        is_iid = config.data.iid
        num_clients = config.clients.total
        labels = loader.labels

        if not is_iid:
            dist, __ = {
                "uniform": dists.uniform,
                "normal": dists.normal
            }[self.config.clients.label_distribution](num_clients, len(labels))
            random.shuffle(dist)

        logging.info('Initializing client data...')

        if not is_iid:
            if self.config.data.bias:
                pref = random.choices(labels, dist)[0]

        logging.info('Total number of clients: %s', num_clients)

        if config.loader == 'shard':
            loader.create_shards()

        # Get data partition size
        if config.loader != 'shard':
            if self.config.data.partition_size:
                partition_size = self.config.data.partition_size

        # Extract data partition for client
        if config.loader == 'iid':
            self.data = loader.get_partition(partition_size)
        elif loader == 'bias':
            self.data = loader.get_partition(partition_size, pref)
        elif loader == 'shard':
            self.data = loader.get_partition()
        else:
            logging.critical('Unknown data loader type.')

        # Extract test parameter settings from the configuration
        test_partition = config.clients.test_partition

        # Extract the trainset and testset if local testing is needed
        if self.config.clients.do_test:
            self.trainset = self.data[:int(len(self.data) * (1 - test_partition))]
            self.testset = self.data[int(len(self.data) * (1 - test_partition)):]
        else:
            self.trainset = self.data


    def train(self):
        """The machine learning training workload on a client."""
        logging.info('Training on client #%s', self.client_id)

        # Perform model training
        trainer.train(self.model, self.trainset, self.config)

        # Extract model weights and biases
        weights = trainer.extract_weights(self.model)

        # Generate a report for the server, performing model testing if applicable
        if self.config.clients.do_test:
            accuracy = trainer.test(self.model, self.testset, 1000)
        else:
            accuracy = 0

        return Report(self.client_id, len(self.data), weights, accuracy)
