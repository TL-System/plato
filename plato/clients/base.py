"""
The base class for all federated learning clients on edge devices or edge servers.
"""

import logging
import os
import pickle
import re
import sys
import uuid
from abc import abstractmethod
from typing import Iterable, Optional

import numpy as np

from plato.callbacks.client import LogProgressCallback
from plato.callbacks.handler import CallbackHandler
from plato.clients.composable import ComposableClient
from plato.clients.strategies import ClientContext
from plato.clients.strategies.legacy import (
    LegacyCommunicationStrategy,
    LegacyLifecycleStrategy,
    LegacyPayloadStrategy,
    LegacyReportingStrategy,
    LegacyTrainingStrategy,
)
from plato.config import Config


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

        # Shared context bridging legacy attributes and strategy-based runtime
        self._context = ClientContext()
        self._context.owner = self
        self._context.client_id = self.client_id
        self._context.current_round = self.current_round
        self._context.comm_simulation = self.comm_simulation
        self._context.chunks = self.chunks
        self._context.callback_handler = self.callback_handler
        self._context.server_payload = self.server_payload
        self._context.processing_time = self.processing_time

        # Strategy adapters preserving legacy hooks
        self._configure_composable(
            lifecycle_strategy=LegacyLifecycleStrategy(self),
            payload_strategy=LegacyPayloadStrategy(self),
            training_strategy=LegacyTrainingStrategy(self),
            reporting_strategy=LegacyReportingStrategy(self),
            communication_strategy=LegacyCommunicationStrategy(self),
        )

    def __repr__(self):
        return f"Client #{self.client_id}"

    def _configure_composable(
        self,
        *,
        lifecycle_strategy,
        payload_strategy,
        training_strategy,
        reporting_strategy,
        communication_strategy,
    ) -> None:
        """Attach strategies and rebuild the composable client runtime."""

        if hasattr(self, "_composable_strategies"):
            for strategy in self._composable_strategies:
                try:
                    strategy.teardown(self._context)
                except Exception:  # pragma: no cover - defensive cleanup
                    logging.debug("Failed to teardown strategy %s.", strategy)

        self.lifecycle_strategy = lifecycle_strategy
        self.payload_strategy = payload_strategy
        self.training_strategy = training_strategy
        self.reporting_strategy = reporting_strategy
        self.communication_strategy = communication_strategy

        self._composable_strategies = (
            self.lifecycle_strategy,
            self.payload_strategy,
            self.training_strategy,
            self.reporting_strategy,
            self.communication_strategy,
        )

        self._composable = ComposableClient(
            owner=self,
            context=self._context,
            lifecycle_strategy=self.lifecycle_strategy,
            payload_strategy=self.payload_strategy,
            training_strategy=self.training_strategy,
            reporting_strategy=self.reporting_strategy,
            communication_strategy=self.communication_strategy,
        )
        self._composable_configured = True

    def _sync_to_context(self, attrs: Optional[Iterable[str]] = None) -> None:
        """Propagate selected owner attributes to the shared context."""
        if attrs is None:
            attrs = self._composable._SYNC_ATTRS
        self._composable._sync_context_from_owner(attrs)

    def _sync_from_context(self, attrs: Optional[Iterable[str]] = None) -> None:
        """Propagate selected context attributes back to the owner."""
        if attrs is None:
            attrs = self._composable._SYNC_ATTRS
        self._composable._sync_owner_from_context(attrs)

    async def start_client(self) -> None:
        """Startup function for a client."""
        await self._composable.start_client()

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
        await self._composable.on_payload_to_arrive(response)

    async def _handle_payload(self, inbound_payload):
        """Handles the inbound payload upon receiving it from the server."""
        await self._composable.handle_server_payload(inbound_payload)

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
        await self._composable.on_chunk(data)

    async def _request_update(self, data) -> None:
        """Upon receiving a request for an urgent model update."""
        await self._composable.on_request_update(data)

    async def _payload_arrived(self, client_id) -> None:
        """Upon receiving a portion of the new payload from the server."""
        await self._composable.on_payload_arrived(client_id)

    async def _payload_done(self, client_id, s3_key=None) -> None:
        """Upon receiving all the new payload from the server."""
        await self._composable.on_payload_done(client_id, s3_key=s3_key)

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
        self._context.callback_handler = self.callback_handler

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
    async def _obtain_model_at_time(self, client_id, requested_time):
        """Retrieving a model update corresponding to a particular wall clock time.

        This method is called during asynchronous training when the server requests
        a model update at a specific wall-clock time.
        """
