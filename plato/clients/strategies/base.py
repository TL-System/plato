"""
Base strategy interfaces for the composable client architecture.

This module defines the shared `ClientContext` container and the abstract
strategy interfaces that will back the redesigned client API. Strategies mirror
the existing lifecycle of `plato.clients.base.Client` but decouple behaviour
into composable units. A future `ComposableClient` will orchestrate these
strategies to reproduce the current inheritance-based implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class ClientContext:
    """
    Shared context passed between client strategies during execution.

    The context allows strategies to:
    - Access common client state (IDs, trainer, datasource, processors, etc.)
    - Share transient data via the `state` dictionary
    - Exchange communication metadata such as socket clients or timers

    Attributes capture the state managed today by `plato.clients.base.Client`
    and `plato.clients.simple.Client`. They will be populated by the upcoming
    composable client implementation.
    """

    def __init__(self) -> None:
        self.owner: Optional[Any] = None
        self.client_id: int = 0
        self.current_round: int = 0

        # Data and learning stack
        self.datasource: Any = None
        self.custom_datasource: Optional[Any] = None
        self.trainer: Any = None
        self.custom_trainer: Optional[Any] = None
        self.trainer_callbacks: Optional[Any] = None
        self.algorithm: Any = None
        self.custom_algorithm: Optional[Any] = None
        self.model: Any = None
        self.custom_model: Optional[Any] = None

        # Dataset partitions
        self.trainset: Any = None
        self.testset: Any = None
        self.sampler: Any = None
        self.testset_sampler: Any = None

        # Payload processors and callbacks
        self.outbound_processor: Any = None
        self.inbound_processor: Any = None
        self.callback_handler: Any = None
        self.reporting_callback: Optional[Any] = None
        self.report_customizer: Optional[Any] = None

        # Communication artefacts
        self.comm_simulation: bool = True
        self.sio: Any = None
        self.s3_client: Any = None
        self.server_payload: Any = None
        self.chunks: list[bytes] = []

        # Runtime measurements
        self.processing_time: float = 0.0
        self.latest_report: Any = None

        # Shared dictionaries for strategies
        self.state: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.timers: Dict[str, float] = {}

    def __repr__(self) -> str:
        """Return a readable identifier for logging/debugging."""
        if self.owner is not None:
            return repr(self.owner)
        return f"ClientContext(client_id={self.client_id}, round={self.current_round})"


class ClientStrategy(ABC):
    """
    Base class for all client strategies.

    Strategies may override `setup`/`teardown` for custom initialisation or
    cleanup, mirroring the behaviour of trainer strategies.
    """

    def setup(self, context: ClientContext) -> None:
        """Optional hook executed when the strategy is attached to a client."""

    def teardown(self, context: ClientContext) -> None:
        """Optional hook executed when the client lifecycle finishes."""


class LifecycleStrategy(ClientStrategy):
    """Strategy interface governing configure/load/allocate stages."""

    @abstractmethod
    def process_server_response(
        self, context: ClientContext, server_response: Dict[str, Any]
    ) -> None:
        """Apply client-specific processing to a server response."""

    @abstractmethod
    def load_data(self, context: ClientContext) -> None:
        """Generate or reload the client's datasource."""

    @abstractmethod
    def configure(self, context: ClientContext) -> None:
        """Construct trainers, algorithms, processors, and callbacks."""

    @abstractmethod
    def allocate_data(self, context: ClientContext) -> None:
        """Assign train/test partitions to the client."""


class PayloadStrategy(ClientStrategy):
    """Strategy interface for inbound/outbound payload orchestration."""

    def reset_payload(self, context: ClientContext) -> None:
        """Reset buffers before receiving a new server payload."""
        context.server_payload = None
        context.chunks.clear()
        context.processing_time = 0.0

    def inbound_received(self, context: ClientContext) -> None:
        """Hook executed before inbound processors begin."""

    def outbound_ready(
        self,
        context: ClientContext,
        report: Any,
        outbound_payload: Any,
    ) -> None:
        """Hook executed before outbound processors begin."""

    async def accumulate_chunk(self, context: ClientContext, chunk: bytes) -> None:
        """Append an inbound payload chunk to the local buffer."""
        context.chunks.append(chunk)

    @abstractmethod
    async def commit_chunk_group(
        self,
        context: ClientContext,
        client_id: int,
    ) -> None:
        """Commit buffered chunks into the assembled payload container."""

    @abstractmethod
    async def finalise_inbound_payload(
        self,
        context: ClientContext,
        client_id: int,
        *,
        s3_key: Optional[str] = None,
    ) -> Any:
        """
        Complete inbound payload assembly, returning the reconstructed payload.
        """

    @abstractmethod
    async def handle_server_payload(
        self,
        context: ClientContext,
        server_payload: Any,
        *,
        training: "TrainingStrategy",
        reporting: "ReportingStrategy",
        communication: "CommunicationStrategy",
    ) -> None:
        """Full inbound processing pipeline ending with outbound transmission."""


class TrainingStrategy(ClientStrategy):
    """Strategy interface for payload loading and training execution."""

    @abstractmethod
    def load_payload(self, context: ClientContext, server_payload: Any) -> None:
        """Load processed server payload onto the client."""

    @abstractmethod
    async def train(self, context: ClientContext) -> Tuple[Any, Any]:
        """Run local training and return (report, outbound_payload)."""


class ReportingStrategy(ClientStrategy):
    """Strategy interface for report construction and async retrieval."""

    @abstractmethod
    def build_report(self, context: ClientContext, report: Any) -> Any:
        """Finalise the report produced after training."""

    @abstractmethod
    async def obtain_model_at_time(
        self, context: ClientContext, client_id: int, requested_time: float
    ) -> Tuple[Any, Any]:
        """Return (report, payload) for asynchronous update requests."""


class CommunicationStrategy(ClientStrategy):
    """Strategy interface for sending data to the server."""

    async def send_report(self, context: ClientContext, report: Any) -> None:
        """Send the client report to the server."""
        raise NotImplementedError

    async def send_payload(self, context: ClientContext, payload: Any) -> None:
        """Send the client payload to the server."""
        raise NotImplementedError

    async def send_report_and_payload(
        self, context: ClientContext, report: Any, payload: Any
    ) -> None:
        """Send both report metadata and payload to the server."""
        await self.send_report(context, report)
        await self.send_payload(context, payload)
