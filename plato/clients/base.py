"""
The base class for all federated learning clients on edge devices or edge servers.
"""

import asyncio
import logging
import os
import pickle
import re
import sys
import time
import uuid
from abc import abstractmethod

import numpy as np
import socketio

from plato.callbacks.client import LogProgressCallback
from plato.callbacks.handler import CallbackHandler
from plato.config import Config
from plato.utils import s3


# pylint: disable=unused-argument, protected-access
class ClientEvents(socketio.AsyncClientNamespace):
    """A custom namespace for socketio.AsyncServer."""

    def __init__(self, namespace, plato_client):
        super().__init__(namespace)
        self.plato_client = plato_client
        self.client_id = plato_client.client_id

    async def on_connect(self):
        """Upon a new connection to the server."""
        logging.info("[Client #%d] Connected to the server.", self.client_id)

    async def on_disconnect(self):
        """Upon a disconnection event."""
        logging.info(
            "[Client #%d] The server disconnected the connection.", self.client_id
        )
        self.plato_client._clear_checkpoint_files()
        os._exit(0)

    async def on_connect_error(self, data):
        """Upon a failed connection attempt to the server."""
        logging.info(
            "[Client #%d] A connection attempt to the server failed.", self.client_id
        )

    async def on_payload_to_arrive(self, data):
        """New payload is about to arrive from the server."""
        await self.plato_client._payload_to_arrive(data["response"])

    async def on_request_update(self, data):
        """The server is requesting an urgent model update."""
        await self.plato_client._request_update(data)

    async def on_chunk(self, data):
        """A chunk of data from the server arrived."""
        await self.plato_client._chunk_arrived(data["data"])

    async def on_payload(self, data):
        """A portion of the new payload from the server arrived."""
        await self.plato_client._payload_arrived(data["id"])

    async def on_payload_done(self, data):
        """All of the new payload sent from the server arrived."""
        if "s3_key" in data:
            await self.plato_client._payload_done(data["id"], s3_key=data["s3_key"])
        else:
            await self.plato_client._payload_done(data["id"])


class Client:
    """A basic federated learning client."""

    def __init__(self, callbacks=None) -> None:
        self.client_id = Config().args.id
        self.current_round = 0
        self.sio = None
        self.chunks = []
        self.server_payload = None
        self.s3_client = None
        self.outbound_processor = None
        self.inbound_processor = None
        self.payload = None
        self.report = None

        self.processing_time = 0

        self.comm_simulation = (
            Config().clients.comm_simulation
            if hasattr(Config().clients, "comm_simulation")
            else True
        )

        if hasattr(Config().algorithm, "cross_silo") and not Config().is_edge_server():
            self.edge_server_id = None

            assert hasattr(Config().algorithm, "total_silos")

        # Starting from the default client callback class, add all supplied server callbacks
        self.callbacks = [LogProgressCallback]
        if callbacks is not None:
            self.callbacks.extend(callbacks)
        self.callback_handler = CallbackHandler(self.callbacks)

    def __repr__(self):
        return f"Client #{self.client_id}"

    async def start_client(self) -> None:
        """Startup function for a client."""
        if hasattr(Config().algorithm, "cross_silo") and not Config().is_edge_server():
            # Contact one of the edge servers
            self.edge_server_id = self.get_edge_server_id()

            logging.info(
                "[Client #%d] Contacting Edge Server #%d.",
                self.client_id,
                self.edge_server_id,
            )
        else:
            await asyncio.sleep(5)
            logging.info("[Client #%d] Contacting the server.", self.client_id)

        self.sio = socketio.AsyncClient(reconnection=True)
        self.sio.register_namespace(ClientEvents(namespace="/", plato_client=self))

        if hasattr(Config().server, "s3_endpoint_url"):
            self.s3_client = s3.S3()

        if hasattr(Config().server, "use_https"):
            uri = f"https://{Config().server.address}"
        else:
            uri = f"http://{Config().server.address}"

        if hasattr(Config().server, "port"):
            # If we are not using a production server deployed in the cloud
            if (
                hasattr(Config().algorithm, "cross_silo")
                and not Config().is_edge_server()
            ):
                uri = f"{uri}:{int(Config().server.port) + int(self.edge_server_id)}"
            else:
                uri = f"{uri}:{Config().server.port}"

        logging.info("[%s] Connecting to the server at %s.", self, uri)
        await self.sio.connect(uri, wait_timeout=600)
        await self.sio.emit("client_alive", {"pid": os.getpid(), "id": self.client_id})

        logging.info("[Client #%d] Waiting to be selected.", self.client_id)
        await self.sio.wait()

    def get_edge_server_id(self):
        """Returns the edge server id of the client in cross-silo FL."""
        launched_client_num = (
            min(
                Config().trainer.max_concurrency
                * max(1, Config().gpu_count())
                * Config().algorithm.total_silos,
                Config().clients.per_round,
            )
            if hasattr(Config().trainer, "max_concurrency")
            else Config().clients.per_round
        )

        edges_launched_clients = [
            len(i)
            for i in np.array_split(
                np.arange(launched_client_num), Config().algorithm.total_silos
            )
        ]

        total = 0
        for i, count in enumerate(edges_launched_clients):
            total += count
            if self.client_id <= total:
                return i + 1 + Config().clients.total_clients

    async def _payload_to_arrive(self, response) -> None:
        """Upon receiving a response from the server."""
        self.current_round = response["current_round"]

        # Update (virtual) client id for client, trainer and algorithm
        self.client_id = response["id"]

        logging.info("[Client #%d] Selected by the server.", self.client_id)

        self.process_server_response(response)

        self._load_data()
        self.configure()
        self._allocate_data()

        self.server_payload = None

        if self.comm_simulation:
            payload_filename = response["payload_filename"]
            with open(payload_filename, "rb") as payload_file:
                self.server_payload = pickle.load(payload_file)

            payload_size = sys.getsizeof(pickle.dumps(self.server_payload))

            logging.info(
                "[%s] Received %.2f MB of payload data from the server (simulated).",
                self,
                payload_size / 1024**2,
            )

            await self._handle_payload(self.server_payload)

    async def _handle_payload(self, inbound_payload):
        """Handles the inbound payload upon receiving it from the server."""
        self.inbound_received(self.inbound_processor)
        self.callback_handler.call_event(
            "on_inbound_received", self, self.inbound_processor
        )

        tic = time.perf_counter()
        processed_inbound_payload = self.inbound_processor.process(inbound_payload)
        self.processing_time = time.perf_counter() - tic

        # Inbound data is processed, computing outbound response
        report, outbound_payload = await self.inbound_processed(
            processed_inbound_payload
        )
        self.callback_handler.call_event(
            "on_inbound_processed", self, processed_inbound_payload
        )

        # Outbound data is ready to be processed
        tic = time.perf_counter()
        self.outbound_ready(report, self.outbound_processor)
        self.callback_handler.call_event(
            "on_outbound_ready", self, report, self.outbound_processor
        )
        processed_outbound_payload = self.outbound_processor.process(outbound_payload)
        self.processing_time += time.perf_counter() - tic
        report.processing_time = self.processing_time

        # Sending the client report as metadata to the server (payload to follow)
        await self.sio.emit(
            "client_report", {"id": self.client_id, "report": pickle.dumps(report)}
        )

        # Sending the client training payload to the server
        await self._send(processed_outbound_payload)

    def inbound_received(self, inbound_processor):
        """
        Override this method to complete additional tasks before the inbound processors start to
        process the data received from the server.
        """

    async def inbound_processed(self, processed_inbound_payload):
        """
        Override this method to conduct customized operations to generate a client's response to
        the server when inbound payload from the server has been processed.
        """
        report, outbound_payload = await self._start_training(processed_inbound_payload)
        return report, outbound_payload

    def outbound_ready(self, report, outbound_processor):
        """
        Override this method to complete additional tasks before the outbound processors start
        to process the data to be sent to the server.
        """

    async def _chunk_arrived(self, data) -> None:
        """Upon receiving a chunk of data from the server."""
        self.chunks.append(data)

    async def _request_update(self, data) -> None:
        """Upon receiving a request for an urgent model update."""
        logging.info(
            "[Client #%s] Urgent request received for model update at time %s.",
            data["client_id"],
            data["time"],
        )

        report, payload = await self._obtain_model_update(
            client_id=data["client_id"],
            requested_time=data["time"],
        )

        # Process outbound data when necessary
        self.callback_handler.call_event(
            "on_outbound_ready", self, report, self.outbound_processor
        )
        self.outbound_ready(report, self.outbound_processor)
        payload = self.outbound_processor.process(payload)

        # Sending the client report as metadata to the server (payload to follow)
        await self.sio.emit(
            "client_report", {"id": self.client_id, "report": pickle.dumps(report)}
        )

        # Sending the client training payload to the server
        await self._send(payload)

    async def _payload_arrived(self, client_id) -> None:
        """Upon receiving a portion of the new payload from the server."""
        assert client_id == self.client_id

        payload = b"".join(self.chunks)
        _data = pickle.loads(payload)
        self.chunks = []

        if self.server_payload is None:
            self.server_payload = _data
        elif isinstance(self.server_payload, list):
            self.server_payload.append(_data)
        else:
            self.server_payload = [self.server_payload]
            self.server_payload.append(_data)

    async def _payload_done(self, client_id, s3_key=None) -> None:
        """Upon receiving all the new payload from the server."""
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
            client_id,
            payload_size / 1024**2,
        )

        await self._handle_payload(self.server_payload)

    async def _start_training(self, inbound_payload):
        """Complete one round of training on this client."""
        self._load_payload(inbound_payload)

        report, outbound_payload = await self._train()

        if Config().is_edge_server():
            logging.info(
                "[Server #%d] Model aggregated on edge server (%s).", os.getpid(), self
            )
        else:
            logging.info("[%s] Model trained.", self)

        return report, outbound_payload

    async def _send_in_chunks(self, data) -> None:
        """Sending a bytes object in fixed-sized chunks to the client."""
        step = 1024**2
        chunks = [data[i : i + step] for i in range(0, len(data), step)]

        for chunk in chunks:
            await self.sio.emit("chunk", {"data": chunk})

        await self.sio.emit("client_payload", {"id": self.client_id})

    async def _send(self, payload) -> None:
        """Sending the client payload to the server using simulation, S3 or socket.io."""
        if self.comm_simulation:
            # If we are using the filesystem to simulate communication over a network
            model_name = (
                Config().trainer.model_name
                if hasattr(Config().trainer, "model_name")
                else "custom"
            )
            if "/" in model_name:
                model_name = model_name.replace("/", "_")
            checkpoint_path = Config().params["checkpoint_path"]
            payload_filename = (
                f"{checkpoint_path}/{model_name}_client_{self.client_id}.pth"
            )
            with open(payload_filename, "wb") as payload_file:
                pickle.dump(payload, payload_file)

            data_size = sys.getsizeof(pickle.dumps(payload))

            logging.info(
                "[%s] Sent %.2f MB of payload data to the server (simulated).",
                self,
                data_size / 1024**2,
            )

        else:
            metadata = {"id": self.client_id}

            if self.s3_client is not None:
                unique_key = uuid.uuid4().hex[:6].upper()
                s3_key = f"client_payload_{self.client_id}_{unique_key}"
                self.s3_client.send_to_s3(s3_key, payload)
                data_size = sys.getsizeof(pickle.dumps(payload))
                metadata["s3_key"] = s3_key
            else:
                if isinstance(payload, list):
                    data_size: int = 0

                    for data in payload:
                        _data = pickle.dumps(data)
                        await self._send_in_chunks(_data)
                        data_size += sys.getsizeof(_data)
                else:
                    _data = pickle.dumps(payload)
                    await self._send_in_chunks(_data)
                    data_size = sys.getsizeof(_data)

            await self.sio.emit("client_payload_done", metadata)

            logging.info(
                "[%s] Sent %.2f MB of payload data to the server.",
                self,
                data_size / 1024**2,
            )

    def _clear_checkpoint_files(self):
        """Delete all the temporary checkpoint files created by the client."""
        model_path = Config().params["model_path"]
        for filename in os.listdir(model_path):
            split = re.match(
                r"(?P<client_id>\d+)_(?P<epoch>\d+)_(?P<training_time>\d+.\d+).pth",
                filename,
            )
            if split is not None:
                file_path = f"{model_path}/{filename}"
                os.remove(file_path)

    def add_callbacks(self, callbacks):
        """Adds a list of callbacks to the client callback handler."""
        self.callback_handler.add_callbacks(callbacks)

    @abstractmethod
    async def _train(self):
        """The machine learning training workload on a client."""

    @abstractmethod
    def configure(self) -> None:
        """Prepare this client for training."""

    @abstractmethod
    def _load_data(self) -> None:
        """Generating data and loading them onto this client."""

    @abstractmethod
    def _allocate_data(self) -> None:
        """Allocate training or testing dataset of this client."""

    @abstractmethod
    def _load_payload(self, server_payload) -> None:
        """Loading the payload onto this client."""

    def process_server_response(self, server_response) -> None:
        """Additional client-specific processing on the server response."""

    @abstractmethod
    async def _obtain_model_update(self, client_id, requested_time):
        """Retrieving a model update corrsponding to a particular wall clock time."""
