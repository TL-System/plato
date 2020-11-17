"""
A basic federated learning client who sends weight updates to the server.
"""

import logging
import json
import random
import pickle
from abc import abstractmethod
from dataclasses import dataclass
import websockets

from config import Config


@dataclass
class Report:
    """Client report sent to the federated learning server."""
    client_id: str
    num_samples: int
    weights: list
    accuracy: float


class Client:
    """A basic federated learning client."""

    def __init__(self):
        self.client_id = Config().args.id
        self.model = None # Machine learning model
        self.data_loaded = False # is training data already loaded from the disk?

        random.seed()


    async def start_client(self):
        """Startup function for a client."""

        if Config().cross_silo and not Config().args.port:
            # Contact one of the edge servers
            logging.info("Client #%s is contacting one of the edge servers...", self.client_id)
            uri = 'ws://{}:{}'.format(Config().server.address,
                Config().server.port + Config().clients.total_clients
                + int(self.client_id) % Config().cross_silo.total_silos + 1)
        else:
            logging.info("Client #%s is contacting the central server...", self.client_id)
            uri = 'ws://{}:{}'.format(Config().server.address, Config().server.port)

        try:
            async with websockets.connect(uri, ping_interval=None, max_size=2 ** 30) as websocket:
                logging.info("Signing in at the server from client #%s...", self.client_id)
                await websocket.send(json.dumps({'id': self.client_id}))

                while True:
                    logging.info("Client #%s is waiting to be selected...", self.client_id)
                    server_response = await websocket.recv()
                    data = json.loads(server_response)

                    if data['id'] == self.client_id and 'payload' in data:
                        logging.info("Client #%s has been selected and receiving the model...",
                                    self.client_id)
                        server_model = await websocket.recv()

                        self.load_model(pickle.loads(server_model))

                        if not self.data_loaded:
                            self.load_data()

                        report = await self.train()

                        logging.info("Model trained on client #%s.", self.client_id)
                        # Sending client ID as metadata to the server (payload to follow)
                        client_update = {'id': self.client_id, 'payload': True}
                        await websocket.send(json.dumps(client_update))

                        # Sending the client training report to the server as payload
                        await websocket.send(pickle.dumps(report))

        except OSError as exception:
            logging.info("Client #%s: connection to the server failed.", self.client_id)
            logging.error(exception)


    @abstractmethod
    def configure(self):
        """Prepare this client for training."""


    @abstractmethod
    def load_data(self):
        """Generating data and loading them onto this client."""


    @abstractmethod
    def load_model(self, server_model):
        """Loading the model onto this client."""


    @abstractmethod
    async def train(self):
        """The machine learning training workload on a client."""
