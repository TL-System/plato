"""
Composable client runtime that orchestrates strategy-based execution.

The `ComposableClient` coordinates lifecycle, payload, training, reporting,
and communication strategies to reproduce the behaviour of legacy client
implementations while enabling gradual migration to the new composition API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import sys
from typing import Any, Iterable, Optional, Sequence

import socketio

from plato.clients.strategies import (
    ClientContext,
    CommunicationStrategy,
    LifecycleStrategy,
    PayloadStrategy,
    ReportingStrategy,
    TrainingStrategy,
)
from plato.config import Config
from plato.utils import s3

LOGGER = logging.getLogger(__name__)


class ComposableClientEvents(socketio.AsyncClientNamespace):
    """Socket.IO namespace that forwards events to the composable core."""

    def __init__(self, namespace: str, core: "ComposableClient") -> None:
        super().__init__(namespace)
        self.core = core

    async def on_connect(self) -> None:
        owner = self.core.owner
        LOGGER.info("[Client #%d] Connected to the server.", owner.client_id)

    async def on_disconnect(self) -> None:
        owner = self.core.owner
        LOGGER.info(
            "[Client #%d] The server disconnected the connection.", owner.client_id
        )
        owner._clear_checkpoint_files()
        os._exit(0)

    async def on_connect_error(self, data: Any) -> None:
        owner = self.core.owner
        LOGGER.info(
            "[Client #%d] A connection attempt to the server failed.",
            owner.client_id,
        )

    async def on_payload_to_arrive(self, data: dict) -> None:
        await self.core.on_payload_to_arrive(data["response"])

    async def on_request_update(self, data: dict) -> None:
        await self.core.on_request_update(data)

    async def on_chunk(self, data: dict) -> None:
        await self.core.on_chunk(data["data"])

    async def on_payload(self, data: dict) -> None:
        await self.core.on_payload_arrived(data["id"])

    async def on_payload_done(self, data: dict) -> None:
        if "s3_key" in data:
            await self.core.on_payload_done(data["id"], s3_key=data["s3_key"])
        else:
            await self.core.on_payload_done(data["id"])


class ComposableClient:
    """
    Core engine for the composable client runtime.

    The engine maintains a shared `ClientContext`, keeps it in sync with the
    legacy client instance, and delegates lifecycle events to strategies.
    """

    _SYNC_ATTRS: Sequence[str] = (
        "client_id",
        "current_round",
        "datasource",
        "custom_datasource",
        "trainer",
        "custom_trainer",
        "trainer_callbacks",
        "algorithm",
        "custom_algorithm",
        "model",
        "custom_model",
        "trainset",
        "testset",
        "sampler",
        "testset_sampler",
        "outbound_processor",
        "inbound_processor",
        "callback_handler",
        "comm_simulation",
        "sio",
        "s3_client",
        "server_payload",
        "chunks",
        "processing_time",
        "payload",
        "report",
        "latest_report",
    )

    def __init__(
        self,
        *,
        owner: Any,
        context: Optional[ClientContext] = None,
        lifecycle_strategy: LifecycleStrategy,
        payload_strategy: PayloadStrategy,
        training_strategy: TrainingStrategy,
        reporting_strategy: ReportingStrategy,
        communication_strategy: CommunicationStrategy,
    ) -> None:
        self.owner = owner
        self.context = context or ClientContext()
        self.context.owner = owner

        self.lifecycle_strategy = lifecycle_strategy
        self.payload_strategy = payload_strategy
        self.training_strategy = training_strategy
        self.reporting_strategy = reporting_strategy
        self.communication_strategy = communication_strategy

        self._strategies: Iterable[Any] = (
            self.lifecycle_strategy,
            self.payload_strategy,
            self.training_strategy,
            self.reporting_strategy,
            self.communication_strategy,
        )

        self._sync_context_from_owner()

        for strategy in self._strategies:
            strategy.setup(self.context)

    def _sync_context_from_owner(self, attrs: Optional[Iterable[str]] = None) -> None:
        """Copy selected attributes from the owner to the shared context."""
        fields = attrs or self._SYNC_ATTRS
        for attr in fields:
            if attr == "chunks":
                setattr(self.context, attr, getattr(self.owner, attr, []))
                continue
            if hasattr(self.owner, attr):
                setattr(self.context, attr, getattr(self.owner, attr))

    def _sync_owner_from_context(self, attrs: Optional[Iterable[str]] = None) -> None:
        """Copy selected attributes from the shared context back to the owner."""
        fields = attrs or self._SYNC_ATTRS
        for attr in fields:
            if hasattr(self.context, attr):
                setattr(self.owner, attr, getattr(self.context, attr))

    async def start_client(self) -> None:
        """Entry point that connects to the server and waits for events."""
        self._sync_context_from_owner()

        if hasattr(Config().algorithm, "cross_silo") and not Config().is_edge_server():
            self.owner.edge_server_id = self.owner.get_edge_server_id()
            LOGGER.info(
                "[Client #%d] Contacting Edge Server #%d.",
                self.owner.client_id,
                self.owner.edge_server_id,
            )
        else:
            await asyncio.sleep(5)
            LOGGER.info("[%s] Contacting the server.", self.owner)

        self.context.sio = socketio.AsyncClient(reconnection=True)
        self.context.sio.register_namespace(
            ComposableClientEvents(namespace="/", core=self)
        )

        if hasattr(Config().server, "s3_endpoint_url"):
            self.context.s3_client = s3.S3()

        if hasattr(Config().server, "use_https"):
            uri = f"https://{Config().server.address}"
        else:
            uri = f"http://{Config().server.address}"

        if hasattr(Config().server, "port"):
            if (
                hasattr(Config().algorithm, "cross_silo")
                and not Config().is_edge_server()
            ):
                uri = f"{uri}:{int(Config().server.port) + int(self.owner.edge_server_id)}"
            else:
                uri = f"{uri}:{Config().server.port}"

        LOGGER.info("[%s] Connecting to the server at %s.", self.owner, uri)
        await self.context.sio.connect(uri, wait_timeout=600)
        await self.context.sio.emit(
            "client_alive", {"pid": os.getpid(), "id": self.context.client_id}
        )

        LOGGER.info("[Client #%d] Waiting to be selected.", self.context.client_id)

        self._sync_owner_from_context()

        await self.context.sio.wait()

    async def on_payload_to_arrive(self, response: dict) -> None:
        """Handle notification that a new payload is about to arrive."""
        self._sync_context_from_owner()

        self.context.current_round = response["current_round"]
        self.context.client_id = response["id"]
        self._sync_owner_from_context(("current_round", "client_id"))

        LOGGER.info("[Client #%d] Selected by the server.", self.owner.client_id)

        self.lifecycle_strategy.process_server_response(self.context, response)
        self._sync_owner_from_context()

        self.lifecycle_strategy.load_data(self.context)
        self._sync_owner_from_context()

        self.lifecycle_strategy.configure(self.context)
        self._sync_owner_from_context()

        self.lifecycle_strategy.allocate_data(self.context)
        self._sync_owner_from_context()

        self.payload_strategy.reset_payload(self.context)
        self._sync_owner_from_context(("server_payload", "processing_time", "chunks"))

        if self.context.comm_simulation:
            payload_filename = response["payload_filename"]
            with open(payload_filename, "rb") as payload_file:
                self.context.server_payload = pickle.load(payload_file)

            payload_size = sys.getsizeof(pickle.dumps(self.context.server_payload))

            LOGGER.info(
                "[%s] Received %.2f MB of payload data from the server (simulated).",
                self.owner,
                payload_size / 1024**2,
            )

            self._sync_owner_from_context(("server_payload",))
            await self.handle_server_payload(self.context.server_payload)
        else:
            self.context.server_payload = None
            self._sync_owner_from_context(("server_payload",))

    async def on_chunk(self, data: bytes) -> None:
        """Handle an incoming payload chunk."""
        self._sync_context_from_owner(("chunks",))
        await self.payload_strategy.accumulate_chunk(self.context, data)
        self._sync_owner_from_context(("chunks",))

    async def on_payload_arrived(self, client_id: int) -> None:
        """Handle completion of a payload part (for multi-part payloads)."""
        await self.payload_strategy.commit_chunk_group(
            self.context, client_id=client_id
        )
        self._sync_owner_from_context(("server_payload", "chunks"))

    async def on_payload_done(
        self, client_id: int, *, s3_key: Optional[str] = None
    ) -> None:
        """Handle completion of the payload transfer."""
        payload = await self.payload_strategy.finalise_inbound_payload(
            self.context, client_id=client_id, s3_key=s3_key
        )
        self._sync_owner_from_context(("server_payload",))
        await self.handle_server_payload(payload)

    async def handle_server_payload(self, server_payload: Any) -> None:
        """Process a fully received server payload."""
        self._sync_context_from_owner()
        await self.payload_strategy.handle_server_payload(
            self.context,
            server_payload,
            training=self.training_strategy,
            reporting=self.reporting_strategy,
            communication=self.communication_strategy,
        )
        self._sync_owner_from_context(
            (
                "processing_time",
                "latest_report",
            )
        )

    async def on_request_update(self, data: dict) -> None:
        """Handle urgent update requests from the server."""
        self._sync_context_from_owner()
        client_id = data["client_id"]
        requested_time = data["time"]

        LOGGER.info(
            "[Client #%s] Urgent request received for model update at time %s.",
            client_id,
            requested_time,
        )

        report, payload = await self.reporting_strategy.obtain_model_at_time(
            self.context, client_id, requested_time
        )

        callbacks = self.context.callback_handler
        outbound_processor = self.context.outbound_processor

        if callbacks is not None:
            callbacks.call_event(
                "on_outbound_ready", self.owner, report, outbound_processor
            )

        self.payload_strategy.outbound_ready(self.context, report, payload)

        if outbound_processor is not None:
            processed_payload = outbound_processor.process(payload)
        else:
            processed_payload = payload

        await self.communication_strategy.send_report_and_payload(
            self.context, report, processed_payload
        )

        self.context.latest_report = report
        self._sync_owner_from_context(("latest_report",))
