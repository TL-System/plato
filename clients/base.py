"""
The base class for all federated learning clients on edge devices or edge servers.
"""

import logging
import os
import pickle
import sys
from abc import abstractmethod

import websockets
from config import Config


class Client:
    """A basic federated learning client."""
    def __init__(self):
        self.client_id = Config().args.id
        self.data_loaded = False  # is training data already loaded from the disk?

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
                int(Config().server.port) +
                int(Config().clients.total_clients) +
                int(self.client_id) % int(Config().algorithm.total_silos) + 1)
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
                await websocket.send(pickle.dumps({'id': self.client_id}))

                while True:
                    logging.info("[Client #%s] Waiting to be selected.",
                                 self.client_id)
                    server_response = await websocket.recv()
                    data = pickle.loads(server_response)

                    if data['id'] == self.client_id:
                        self.process_server_response(data)
                        logging.info("[Client #%s] Selected by the server.",
                                     self.client_id)

                        if not self.data_loaded:
                            self.load_data()

                        if 'payload' in data:

                            server_payload = await self.recv(
                                self.client_id, data, websocket)
                            self.load_payload(server_payload)

                        report, payload = await self.train()

                        if Config().is_edge_server():
                            logging.info(
                                "[Server #%d] Model aggregated on edge server (client #%s).",
                                os.getpid(), self.client_id)
                        else:
                            logging.info("[Client #%s] Model trained.",
                                         self.client_id)

                        # Sending the client report as metadata to the server (payload to follow)
                        client_report = {
                            'id': self.client_id,
                            'report': report,
                            'payload': True
                        }
                        await websocket.send(pickle.dumps(client_report))

                        # Sending the client training payload to the server
                        await self.send(websocket, payload)

        except OSError as exception:
            logging.info("[Client #%s] Connection to the server failed.",
                         self.client_id)
            logging.error(exception)

    async def recv(self, client_id, data, websocket):
        """Receiving the payload from the server using WebSockets."""

        logging.info("[Client #%s] Receiving payload data from the server.",
                     client_id)

        if 'payload_length' in data:
            server_payload = []
            payload_size = 0

            for __ in range(0, data['payload_length']):
                _data = await websocket.recv()
                payload = pickle.loads(_data)
                server_payload.append(payload)
                payload_size += sys.getsizeof(_data)
        else:
            _data = await websocket.recv()
            server_payload = pickle.loads(_data)
            payload_size = sys.getsizeof(_data)

        logging.info(
            "[Client #%s] Received %s bytes of payload data from the server.",
            client_id, payload_size)

        return server_payload

    async def send(self, websocket, payload):
        """Sending the client payload to the server using WebSockets."""
        if isinstance(payload, list):
            data_size = 0

            for data in payload:
                _data = pickle.dumps(data)
                await websocket.send(_data)
                data_size += sys.getsizeof(_data)
        else:
            _data = pickle.dumps(payload)
            await websocket.send(_data)
            data_size = sys.getsizeof(_data)

        logging.info(
            "[Client #%s] Sent %s bytes of payload data to the server.",
            self.client_id, data_size)

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
