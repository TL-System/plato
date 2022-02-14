"""
The base class for all federated learning clients on edge devices or edge servers.
"""

import asyncio
import logging
import os
import pickle
import re
import sys
import uuid
from abc import abstractmethod
from dataclasses import dataclass

import socketio
from plato.config import Config
from plato.utils import s3


@dataclass
class Report:
    """Client report, to be sent to the federated learning server."""
    num_samples: int
    accuracy: float
    training_time: float


class ClientEvents(socketio.AsyncClientNamespace):
    """ A custom namespace for socketio.AsyncServer. """

    def __init__(self, namespace, plato_client):
        super().__init__(namespace)
        self.plato_client = plato_client
        self.client_id = plato_client.client_id

    #pylint: disable=unused-argument
    async def on_connect(self):
        """ Upon a new connection to the server. """
        logging.info("[Client #%d] Connected to the server.", self.client_id)

    # pylint: disable=protected-access
    async def on_disconnect(self):
        """ Upon a disconnection event. """
        logging.info("[Client #%d] The server disconnected the connection.",
                     self.client_id)
        self.plato_client.clear_checkpoint_files()
        os._exit(0)

    async def on_connect_error(self, data):
        """ Upon a failed connection attempt to the server. """
        logging.info("[Client #%d] A connection attempt to the server failed.",
                     self.client_id)

    async def on_payload_to_arrive(self, data):
        """ New payload is about to arrive from the server. """
        await self.plato_client.payload_to_arrive(data['response'])

    async def on_request_update(self, data):
        """ The server is requesting an urgent model update. """
        await self.plato_client.request_update(data)

    async def on_chunk(self, data):
        """ A chunk of data from the server arrived. """
        await self.plato_client.chunk_arrived(data['data'])

    async def on_payload(self, data):
        """ A portion of the new payload from the server arrived. """
        await self.plato_client.payload_arrived(data['id'])

    async def on_payload_done(self, data):
        """ All of the new payload sent from the server arrived. """
        if 's3_key' in data:
            await self.plato_client.payload_done(data['id'],
                                                 s3_key=data['s3_key'])
        else:
            await self.plato_client.payload_done(data['id'])


class Client:
    """ A basic federated learning client. """

    def __init__(self) -> None:
        self.client_id = Config().args.id
        self.sio = None
        self.chunks = []
        self.server_payload = None
        self.data_loaded = False  # is training data already loaded from the disk?
        self.s3_client = None
        self.outbound_processor = None
        self.inbound_processor = None

        self.comm_simulation = False
        if hasattr(Config().clients,
                   'comm_simulation') and Config().clients.comm_simulation:
            self.comm_simulation = True

        if hasattr(Config().algorithm,
                   'cross_silo') and not Config().is_edge_server():
            self.edge_server_id = None

            assert hasattr(Config().algorithm, 'total_silos')

    def __repr__(self):
        return f'Client #{self.client_id}'

    async def start_client(self) -> None:
        """ Startup function for a client. """

        if hasattr(Config().algorithm,
                   'cross_silo') and not Config().is_edge_server():
            # Contact one of the edge servers
            if hasattr(Config().clients,
                       'simulation') and Config().clients.simulation:
                self.edge_server_id = int(
                    Config().clients.per_round) + (self.client_id - 1) % int(
                        Config().algorithm.total_silos) + 1
            else:
                self.edge_server_id = int(Config().clients.total_clients) + (
                    self.client_id - 1) % int(
                        Config().algorithm.total_silos) + 1
            logging.info("[Client #%d] Contacting Edge server #%d.",
                         self.client_id, self.edge_server_id)
        else:
            await asyncio.sleep(5)
            logging.info("[Client #%d] Contacting the server.", self.client_id)

        self.sio = socketio.AsyncClient(reconnection=True)
        self.sio.register_namespace(
            ClientEvents(namespace='/', plato_client=self))

        if hasattr(Config().server, 's3_endpoint_url'):
            self.s3_client = s3.S3()

        if hasattr(Config().server, 'use_https'):
            uri = f'https://{Config().server.address}'
        else:
            uri = f'http://{Config().server.address}'

        if hasattr(Config().server, 'port'):
            # If we are not using a production server deployed in the cloud
            if hasattr(Config().algorithm,
                       'cross_silo') and not Config().is_edge_server():
                uri = f'{uri}:{int(Config().server.port) + int(self.edge_server_id)}'
            else:
                uri = f'{uri}:{Config().server.port}'

        logging.info("[%s] Connecting to the server at %s.", self, uri)
        await self.sio.connect(uri, wait_timeout=600)
        await self.sio.emit('client_alive', {'id': self.client_id})

        logging.info("[Client #%d] Waiting to be selected.", self.client_id)
        await self.sio.wait()

    async def payload_to_arrive(self, response) -> None:
        """ Upon receiving a response from the server. """
        self.process_server_response(response)

        # Update (virtual) client id for client, trainer and algorithm
        if hasattr(Config().clients,
                   'simulation') and Config().clients.simulation:
            self.client_id = response['id']
            self.configure()

        logging.info("[Client #%d] Selected by the server.", self.client_id)

        if (hasattr(Config().data, 'reload_data')
                and Config().data.reload_data) or not self.data_loaded:
            self.load_data()

        if hasattr(Config().clients,
                   'comm_simulation') and Config().clients.comm_simulation:
            payload_filename = response['payload_filename']
            with open(payload_filename, 'rb') as payload_file:
                self.server_payload = pickle.load(payload_file)

            payload_size = sys.getsizeof(pickle.dumps(self.server_payload))

            logging.info(
                "[%s] Received %.2f MB of payload data from the server (simulated).",
                self, payload_size / 1024**2)

            self.server_payload = self.inbound_processor.process(
                self.server_payload)

            await self.start_training()

    async def chunk_arrived(self, data) -> None:
        """ Upon receiving a chunk of data from the server. """
        self.chunks.append(data)

    async def request_update(self, data) -> None:
        """ Upon receiving a request for an urgent model update. """
        logging.info(
            "[Client #%s] Urgent request received for model update at time %s.",
            self.client_id, data['time'])

        report, payload = await self.obtain_model_update(data['time'])

        # Sending the client report as metadata to the server (payload to follow)
        await self.sio.emit('client_report', {
            'id': self.client_id,
            'report': pickle.dumps(report)
        })

        # Sending the client training payload to the server
        await self.send(payload)

    async def payload_arrived(self, client_id) -> None:
        """ Upon receiving a portion of the new payload from the server. """
        assert client_id == self.client_id

        payload = b''.join(self.chunks)
        _data = pickle.loads(payload)
        self.chunks = []

        if self.server_payload is None:
            self.server_payload = _data
        elif isinstance(self.server_payload, list):
            self.server_payload.append(_data)
        else:
            self.server_payload = [self.server_payload]
            self.server_payload.append(_data)

    async def payload_done(self, client_id, s3_key=None) -> None:
        """ Upon receiving all the new payload from the server. """
        payload_size = 0

        if s3_key is None:
            if isinstance(self.server_payload, list):
                for _data in self.server_payload:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            elif isinstance(self.server_payload, dict):
                for key, value in self.server_payload.items():
                    payload_size += sys.getsizeof(pickle.dumps({key: value}))
            else:
                payload_size = sys.getsizeof(pickle.dumps(self.server_payload))
        else:
            self.server_payload = self.s3_client.receive_from_s3(s3_key)
            payload_size = sys.getsizeof(pickle.dumps(self.server_payload))

        assert client_id == self.client_id

        logging.info(
            "[Client #%d] Received %.2f MB of payload data from the server.",
            client_id, payload_size / 1024**2)

        self.server_payload = self.inbound_processor.process(
            self.server_payload)

        await self.start_training()

    async def start_training(self):
        """ Complete one round of training on this client. """
        self.load_payload(self.server_payload)
        self.server_payload = None

        report, payload = await self.train()

        if Config().is_edge_server():
            logging.info("[Server #%d] Model aggregated on edge server (%s).",
                         os.getpid(), self)
        else:
            logging.info("[%s] Model trained.", self)

        # Sending the client report as metadata to the server (payload to follow)
        await self.sio.emit('client_report', {
            'id': self.client_id,
            'report': pickle.dumps(report)
        })

        # Sending the client training payload to the server
        await self.send(payload)

    async def send_in_chunks(self, data) -> None:
        """ Sending a bytes object in fixed-sized chunks to the client. """
        step = 1024 ^ 2
        chunks = [data[i:i + step] for i in range(0, len(data), step)]

        for chunk in chunks:
            await self.sio.emit('chunk', {'data': chunk})

        await self.sio.emit('client_payload', {'id': self.client_id})

    async def send(self, payload) -> None:
        """Sending the client payload to the server using simulation, S3 or socket.io."""
        # First apply outbound processors, if any
        payload = self.outbound_processor.process(payload)

        if self.comm_simulation:
            # If we are using the filesystem to simulate communication over a network
            model_name = Config().trainer.model_name if hasattr(
                Config().trainer, 'model_name') else 'custom'
            checkpoint_dir = Config().params['checkpoint_dir']
            payload_filename = f"{checkpoint_dir}/{model_name}_client_{self.client_id}.pth"
            with open(payload_filename, 'wb') as payload_file:
                pickle.dump(payload, payload_file)

            logging.info(
                "[%s] Sent %.2f MB of payload data to the server (simulated).",
                self,
                sys.getsizeof(pickle.dumps(payload)) / 1024**2)
        else:
            metadata = {'id': self.client_id}

            if self.s3_client is not None:
                unique_key = uuid.uuid4().hex[:6].upper()
                s3_key = f'client_payload_{self.client_id}_{unique_key}'
                self.s3_client.send_to_s3(s3_key, payload)
                data_size = sys.getsizeof(pickle.dumps(payload))
                metadata['s3_key'] = s3_key
            else:
                if isinstance(payload, list):
                    data_size: int = 0

                    for data in payload:
                        _data = pickle.dumps(data)
                        await self.send_in_chunks(_data)
                        data_size += sys.getsizeof(_data)
                else:
                    _data = pickle.dumps(payload)
                    await self.send_in_chunks(_data)
                    data_size = sys.getsizeof(_data)

            await self.sio.emit('client_payload_done', metadata)

            logging.info("[%s] Sent %.2f MB of payload data to the server.",
                         self, data_size / 1024**2)

    def process_server_response(self, server_response) -> None:
        """Additional client-specific processing on the server response."""

    def clear_checkpoint_files(self):
        """ Delete all the temporary checkpoint files created by the client. """
        if hasattr(Config().server,
                   'request_update') and Config().server.request_update:
            model_dir = Config().params['model_dir']
            for filename in os.listdir(model_dir):
                split = re.match(
                    r"(?P<client_id>\d+)_(?P<epoch>\d+)_(?P<training_time>\d+.\d+).pth",
                    filename)
                if split is not None and self.client_id == int(
                        split.group('client_id')):
                    file_path = f'{model_dir}/{filename}'
                    os.remove(file_path)

    @abstractmethod
    def configure(self) -> None:
        """ Prepare this client for training. """

    @abstractmethod
    def load_data(self) -> None:
        """ Generating data and loading them onto this client. """

    @abstractmethod
    def save_model(self, model_checkpoint) -> None:
        """ Saving the model to a model checkpoint. """

    @abstractmethod
    def load_model(self, model_checkpoint) -> None:
        """ Loading the model from a model checkpoint. """

    @abstractmethod
    def load_payload(self, server_payload) -> None:
        """ Loading the payload onto this client. """

    @abstractmethod
    async def train(self):
        """ The machine learning training workload on a client. """

    @abstractmethod
    async def obtain_model_update(self, wall_time):
        """ Retrieving a model update corrsponding to a particular wall clock time. """
