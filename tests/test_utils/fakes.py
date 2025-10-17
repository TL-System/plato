"""
Reusable fake components for exercising client and server lifecycles in tests.

These lightweight implementations keep unit tests fast while providing just
enough behaviour to drive the composable stacks (clients, servers, trainers).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional

import torch
from torch.utils.data import TensorDataset

from plato.clients.strategies.base import (
    ClientContext,
    CommunicationStrategy,
    LifecycleStrategy,
    PayloadStrategy,
    ReportingStrategy,
    TrainingStrategy,
)
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FakeModel(torch.nn.Module):
    """Small linear model that keeps forward passes deterministic for tests."""

    def __init__(self, input_dim: int = 4, num_classes: int = 2):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        with torch.no_grad():
            torch.manual_seed(0)
            self.linear.weight.uniform_(-0.01, 0.01)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


@dataclass
class FakeDatasource:
    """
    Datasource producing deterministic TensorDataset partitions for tests.

    The class mimics the minimal API used by lifecycle strategies: methods to
    report the number of training examples and to fetch train/test datasets.
    """

    train_length: int = 8
    test_length: int = 4
    input_dim: int = 4
    num_classes: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        generator = torch.Generator().manual_seed(self.seed)

        total_length = self.train_length + self.test_length
        features = torch.randn(total_length, self.input_dim, generator=generator)
        labels = torch.randint(
            0,
            self.num_classes,
            (total_length,),
            generator=generator,
        )

        train_slice = slice(0, self.train_length)
        test_slice = slice(self.train_length, total_length)

        self._train = TensorDataset(features[train_slice], labels[train_slice])
        self._test = TensorDataset(features[test_slice], labels[test_slice])

    def num_train_examples(self) -> int:
        return len(self._train)

    def get_train_set(self):
        return self._train

    def get_test_set(self):
        return self._test


class StaticTrainingStrategy(TrainingStrategy):
    """
    Training strategy that skips optimisation and returns canned results.

    Useful for testing client lifecycles without running real training loops.
    """

    def __init__(self, payload_template: Optional[Dict[str, torch.Tensor]] = None):
        self._payload_template = payload_template or {"weights": torch.zeros(1)}

    def load_payload(self, context: ClientContext, server_payload) -> None:
        context.state["loaded_payload"] = server_payload

    async def train(self, context: ClientContext):
        trainset = getattr(context, "trainset", None)
        num_samples = len(trainset) if trainset is not None else 0
        report = SimpleNamespace(
            client_id=context.client_id,
            num_samples=num_samples,
        )
        payload = {
            name: tensor.clone() for name, tensor in self._payload_template.items()
        }
        context.state["last_report"] = report
        context.state["last_payload"] = payload
        return report, payload


class IdentityLifecycleStrategy(LifecycleStrategy):
    """
    Lifecycle strategy that injects pre-specified components into the context.

    Each stage simply copies provided callables/objects into the context, which
    allows tests to focus on orchestration logic instead of complex setup.
    """

    def __init__(
        self,
        datasource_factory=None,
        trainer_factory=None,
        algorithm_factory=None,
    ):
        self._datasource_factory = datasource_factory
        self._trainer_factory = trainer_factory
        self._algorithm_factory = algorithm_factory

    def process_server_response(self, context: ClientContext, server_response):
        context.state["last_server_response"] = server_response

    def load_data(self, context: ClientContext) -> None:
        if self._datasource_factory is None:
            return
        context.datasource = self._datasource_factory()

    def configure(self, context: ClientContext) -> None:
        if context.model is None and hasattr(context, "custom_model"):
            context.model = context.custom_model

        if context.trainer is None and self._trainer_factory is not None:
            context.trainer = self._trainer_factory()

        if context.algorithm is None and self._algorithm_factory is not None:
            context.algorithm = self._algorithm_factory()

    def allocate_data(self, context: ClientContext) -> None:
        if context.datasource is None:
            return
        context.trainset = context.datasource.get_train_set()
        context.testset = context.datasource.get_test_set()


class InMemoryReportingStrategy(ReportingStrategy):
    """Reporting strategy that stores the latest report for assertions."""

    def build_report(self, context: ClientContext, report):
        context.state["built_report"] = report
        customiser = getattr(context, "report_customizer", None)
        return customiser(report) if callable(customiser) else report

    async def obtain_model_at_time(
        self, context: ClientContext, client_id: int, requested_time: float
    ):
        payload = context.state.get("last_payload")
        report = context.state.get("last_report")
        return report, payload


class NoOpCommunicationStrategy(CommunicationStrategy):
    """Communication strategy that records messages instead of sending them."""

    async def send_report(self, context: ClientContext, report) -> None:
        context.state.setdefault("sent_reports", []).append(report)

    async def send_payload(self, context: ClientContext, payload) -> None:
        context.state.setdefault("sent_payloads", []).append(payload)


class WeightedAverageAggregation(AggregationStrategy):
    """
    Aggregation strategy performing a simple weighted average of client deltas.

    This mirrors the behaviour expected by most server tests without relying on
    the full FedAvg implementation.
    """

    async def aggregate_deltas(self, updates, deltas_received, context: ServerContext):
        if not deltas_received:
            return {}

        total_weight = sum(
            getattr(update.report, "num_samples", 1) for update in updates
        )
        if total_weight == 0:
            total_weight = len(deltas_received)

        aggregated: Dict[str, torch.Tensor] = {}
        for name in deltas_received[0]:
            aggregated[name] = torch.zeros_like(deltas_received[0][name])

        for update, delta in zip(updates, deltas_received):
            weight = getattr(update.report, "num_samples", 1) / total_weight
            for name, value in delta.items():
                aggregated[name] = aggregated[name] + value * weight

        return aggregated


class RecordingPayloadStrategy(PayloadStrategy):
    """
    Payload strategy that records lifecycle events for assertions in tests.

    The implementation mirrors the high-level flow of the default strategy
    while minimising external dependencies so unit tests can introspect
    event ordering.
    """

    def __init__(self) -> None:
        self.events: list[str] = []

    def reset_payload(self, context: ClientContext) -> None:
        super().reset_payload(context)
        self.events.append("reset")
        context.state.setdefault("event_log", []).append("reset")

    def inbound_received(self, context: ClientContext) -> None:
        self.events.append("inbound_received")
        context.state.setdefault("event_log", []).append("inbound_received")

    def outbound_ready(
        self,
        context: ClientContext,
        report,
        outbound_payload,
    ) -> None:
        self.events.append("outbound_ready")
        context.state.setdefault("event_log", []).append("payload_outbound")

    async def accumulate_chunk(self, context: ClientContext, data: bytes) -> None:
        self.events.append("chunk")
        context.state.setdefault("event_log", []).append("chunk")
        await super().accumulate_chunk(context, data)

    async def commit_chunk_group(
        self,
        context: ClientContext,
        client_id: int,
    ) -> None:
        self.events.append("commit")
        context.state.setdefault("event_log", []).append("commit")
        if context.chunks:
            payload_bytes = b"".join(context.chunks)
            context.server_payload = pickle.loads(payload_bytes)
            context.chunks.clear()

    async def finalise_inbound_payload(
        self,
        context: ClientContext,
        client_id: int,
        *,
        s3_key: Optional[str] = None,
    ):
        self.events.append("finalise")
        context.state.setdefault("event_log", []).append("finalise")
        return context.server_payload

    async def handle_server_payload(
        self,
        context: ClientContext,
        server_payload,
        *,
        training: TrainingStrategy,
        reporting: ReportingStrategy,
        communication: CommunicationStrategy,
    ) -> None:
        self.events.append("handle_start")
        context.state.setdefault("event_log", []).append("handle_start")

        owner = context.owner or context
        callbacks = context.callback_handler
        inbound_processor = context.inbound_processor
        outbound_processor = context.outbound_processor

        self.inbound_received(context)

        if callbacks is not None:
            callbacks.call_event("on_inbound_received", owner, inbound_processor)

        if inbound_processor is not None:
            processed_inbound = inbound_processor.process(server_payload)
        else:
            processed_inbound = server_payload

        context.processing_time = 0.0
        training.load_payload(context, processed_inbound)
        report, outbound_payload = await training.train(context)
        context.state["last_train_payload"] = outbound_payload

        if callbacks is not None:
            callbacks.call_event("on_inbound_processed", owner, processed_inbound)

        report = reporting.build_report(context, report)
        context.state["last_report"] = report

        self.outbound_ready(context, report, outbound_payload)

        if outbound_processor is not None:
            outbound_payload = outbound_processor.process(outbound_payload)

        await communication.send_report_and_payload(context, report, outbound_payload)
        context.state["last_payload"] = outbound_payload

        self.events.append("handle_done")
        context.state.setdefault("event_log", []).append("handle_done")
