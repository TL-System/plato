"""
Legacy strategy adapters for the composable client runtime.

These strategies delegate to the existing methods on `plato.clients.base.Client`
instances so that current subclasses continue to operate while the new
composition-based design is introduced incrementally.
"""

from __future__ import annotations

import pickle
import time
from typing import Any, Tuple

from plato.clients.strategies.base import (
    ClientContext,
    CommunicationStrategy,
    LifecycleStrategy,
    PayloadStrategy,
    ReportingStrategy,
    TrainingStrategy,
)
from plato.clients.strategies.defaults import DefaultPayloadStrategy


class LegacyLifecycleStrategy(LifecycleStrategy):
    """Adapter that forwards lifecycle stages to legacy client methods."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def process_server_response(
        self, context: ClientContext, server_response: dict
    ) -> None:
        self.owner.process_server_response(server_response)

    def load_data(self, context: ClientContext) -> None:
        self.owner._load_data()

    def configure(self, context: ClientContext) -> None:
        self.owner.configure()

    def allocate_data(self, context: ClientContext) -> None:
        self.owner._allocate_data()


class LegacyTrainingStrategy(TrainingStrategy):
    """Adapter that reuses the legacy `_load_payload` and `_train` hooks."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def load_payload(self, context: ClientContext, server_payload: Any) -> None:
        self.owner._load_payload(server_payload)

    async def train(self, context: ClientContext) -> Tuple[Any, Any]:
        return await self.owner._train()


class LegacyReportingStrategy(ReportingStrategy):
    """Adapter that preserves legacy report customisation and async retrieval."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def build_report(self, context: ClientContext, report: Any) -> Any:
        context.latest_report = report
        return report

    async def obtain_model_at_time(
        self, context: ClientContext, client_id: int, requested_time: float
    ) -> Tuple[Any, Any]:
        report, payload = await self.owner._obtain_model_at_time(
            client_id, requested_time
        )
        context.latest_report = report
        return report, payload


class LegacyCommunicationStrategy(CommunicationStrategy):
    """Adapter that reuses the legacy payload transmission helpers."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    async def send_report(self, context: ClientContext, report: Any) -> None:
        if context.sio is None:
            raise RuntimeError("Socket client is not initialised.")

        await context.sio.emit(
            "client_report",
            {"id": context.client_id, "report": pickle.dumps(report)},
        )

    async def send_payload(self, context: ClientContext, payload: Any) -> None:
        await self.owner._send(payload)


class LegacyPayloadStrategy(DefaultPayloadStrategy):
    """Adapter that preserves the behaviour of legacy `_handle_payload`."""

    def __init__(self, owner: Any) -> None:
        super().__init__()
        self.owner = owner

    def inbound_received(self, context: ClientContext) -> None:
        inbound_processor = getattr(self.owner, "inbound_processor", None)
        self.owner.inbound_received(inbound_processor)

    def outbound_ready(
        self,
        context: ClientContext,
        report: Any,
        outbound_payload: Any,
    ) -> None:
        outbound_processor = getattr(self.owner, "outbound_processor", None)
        self.owner.outbound_ready(report, outbound_processor)

    async def handle_server_payload(
        self,
        context: ClientContext,
        server_payload: Any,
        *,
        training: TrainingStrategy,
        reporting: ReportingStrategy,
        communication: CommunicationStrategy,
    ) -> None:
        owner = self.owner
        callbacks = context.callback_handler
        inbound_processor = getattr(owner, "inbound_processor", None)
        outbound_processor = getattr(owner, "outbound_processor", None)

        self.inbound_received(context)

        if callbacks is not None:
            callbacks.call_event("on_inbound_received", owner, inbound_processor)

        tic = time.perf_counter()
        processed_inbound = (
            inbound_processor.process(server_payload)
            if inbound_processor is not None
            else server_payload
        )
        context.processing_time = time.perf_counter() - tic

        report, outbound_payload = await owner.inbound_processed(processed_inbound)

        if callbacks is not None:
            callbacks.call_event("on_inbound_processed", owner, processed_inbound)

        report = reporting.build_report(context, report)

        self.outbound_ready(context, report, outbound_payload)

        if callbacks is not None:
            callbacks.call_event("on_outbound_ready", owner, report, outbound_processor)

        tic = time.perf_counter()
        processed_outbound = (
            outbound_processor.process(outbound_payload)
            if outbound_processor is not None
            else outbound_payload
        )
        context.processing_time += time.perf_counter() - tic

        try:
            setattr(report, "processing_time", context.processing_time)
        except AttributeError:
            pass

        await communication.send_report_and_payload(context, report, processed_outbound)
        context.latest_report = report
