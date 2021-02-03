"""
The base class for all federated learning clients on edge devices or edge servers.
"""

import logging
import json
import random
import os
import pickle
from abc import abstractmethod
import websockets

from config import Config


class Client:
    """A basic federated learning client."""
    def __init__(self):
        self.client_id = Config().args.id
        self.model = None  # Machine learning model
        self.data_loaded = False  # is training data already loaded from the disk?

        random.seed()

    async def start_client(self):
        """Startup function for a client."""

        if hasattr(Config().algorithm,
                   'cross_silo') and not Config().is_edge_server():
            # Contact one of the edge servers
            logging.info("[Client #%s] Contacting one of the edge servers.",
                         self.client_id)

            assert hasattr(Config().algorithm, 'total_silos')

            uri = 'ws://{}:{}'.format(
                Config().server.address,
                Config().server.port + Config().clients.total_clients +
                int(self.client_id) % Config().algorithm.total_silos + 1)
        else:
            logging.info("[Client #%s] Contacting the central server.",
                         self.client_id)
            uri = 'ws://{}:{}'.format(Config().server.address,
                                      Config().server.port)

        try:
            async with websockets.connect(uri,
                                          ping_interval=None,
                                          max_size=2**30) as websocket:
                logging.info("[Client #%s] Signing in at the server.",
                             self.client_id)
                await websocket.send(json.dumps({'id': self.client_id}))

                while True:
                    logging.info("[Client #%s] Waiting to be selected.",
                                 self.client_id)
                    server_response = await websocket.recv()
                    data = json.loads(server_response)

                    if data['id'] == self.client_id:
                        self.process_server_response(data)
                        logging.info("[Client #%s] Selected by the server.",
                                     self.client_id)

                        if not self.data_loaded:
                            self.load_data()

                        if 'trainer_counter_file_id' in data:
                            Config().trainer_counter_file = Config(
                            ).trainer_counter_dir + str(
                                data['trainer_counter_file_id'])

                        if 'payload' in data:
                            logging.info(
                                "[Client #%s] Receiving payload from the server.",
                                self.client_id)
                            server_payload = await websocket.recv()

                            self.load_payload(pickle.loads(server_payload))

                        report = await self.train()

                        if Config().is_edge_server():
                            logging.info(
                                "[Server #%d] Model aggregated on edge server (client #%s).",
                                os.getpid(), self.client_id)
                        else:
                            logging.info("[Client #%s] Model trained.",
                                         self.client_id)

                        # Sending client ID as metadata to the server (payload to follow)
                        client_update = {'id': self.client_id, 'payload': True}
                        await websocket.send(json.dumps(client_update))

                        # Sending the client training report to the server as payload
                        logging.info(
                            "[Client #%s] Sending reports to the server.",
                            self.client_id)
                        await websocket.send(pickle.dumps(report))

        except OSError as exception:
            logging.info("[Client #%s] Connection to the server failed.",
                         self.client_id)
            logging.error(exception)

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""

    @abstractmethod
    def configure(self):
        """Prepare this client for training."""

    @abstractmethod
    def load_data(self):
        """Generating data and loading them onto this client."""

    @abstractmethod
    def load_payload(self, server_payload):
        """Loading the payload onto this client."""

    @abstractmethod
    async def train(self):
        """The machine learning training workload on a client."""
