"""
Edge client strategies.

These strategies tailor the default client pipeline for edge servers that
coordinate downstream clients before sending aggregated updates to the
central server.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Tuple

from plato.clients.strategies.base import ClientContext, TrainingStrategy
from plato.clients.strategies.defaults import DefaultLifecycleStrategy
from plato.config import Config
from plato.processors import registry as processor_registry


class EdgeLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that skips local data handling for edge clients."""

    def process_server_response(
        self, context: ClientContext, server_response: dict
    ) -> None:
        """Propagate global round metadata to the edge server."""
        owner = context.owner
        if owner is not None and "current_global_round" in server_response:
            owner.server.current_global_round = server_response["current_global_round"]

    def load_data(self, context: ClientContext) -> None:
        """Edge clients do not load local datasets."""

    def allocate_data(self, context: ClientContext) -> None:
        """Edge clients do not allocate local datasets."""

    def configure(self, context: ClientContext) -> None:
        """Configure processors to use the edge server trainer."""
        super().configure(context)

        owner = context.owner
        if owner is not None:
            outbound, inbound = processor_registry.get(
                "Client", client_id=context.client_id, trainer=owner.server.trainer
            )
            context.outbound_processor = outbound
            context.inbound_processor = inbound


class EdgeTrainingStrategy(TrainingStrategy):
    """Training strategy that orchestrates aggregation on the edge server."""

    def load_payload(self, context: ClientContext, server_payload: Any) -> None:
        owner = context.owner
        if owner is None:
            raise RuntimeError("EdgeTrainingStrategy requires an owning client.")
        owner.server.algorithm.load_weights(server_payload)

    async def train(self, context: ClientContext) -> Tuple[Any, Any]:
        owner = context.owner
        if owner is None:
            raise RuntimeError("EdgeTrainingStrategy requires an owning client.")

        server = owner.server

        training_start_time = time.perf_counter()
        server.new_global_round_begins.set()

        await server.model_aggregated.wait()
        server.model_aggregated.clear()

        weights = server.algorithm.extract_weights()
        average_accuracy = server.average_accuracy
        accuracy = server.accuracy

        if (
            hasattr(Config().clients, "sleep_simulation")
            and Config().clients.sleep_simulation
        ):
            training_time = server.edge_training_time
            server.edge_training_time = 0
        else:
            training_time = time.perf_counter() - training_start_time

        comm_time = time.time()

        edge_server_comm_time = server.edge_comm_time
        server.edge_comm_time = 0

        report = SimpleNamespace(
            client_id=context.client_id,
            num_samples=server.total_samples,
            accuracy=accuracy,
            training_time=training_time,
            comm_time=comm_time,
            update_response=False,
            average_accuracy=average_accuracy,
            edge_server_comm_overhead=server.comm_overhead,
            edge_server_comm_time=edge_server_comm_time,
        )

        customized_report = owner.customize_report(report)
        owner._report = customized_report  # pylint: disable=protected-access
        server.comm_overhead = 0
        context.latest_report = customized_report

        return customized_report, weights
