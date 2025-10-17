"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import logging
from dataclasses import dataclass

from plato.clients import edge
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.edge import EdgeLifecycleStrategy, EdgeTrainingStrategy


@dataclass
class Report(edge.Report):
    """Report from an Axiothea edge server, to be sent to the central server."""


class CsMamlEdgeLifecycleStrategy(EdgeLifecycleStrategy):
    """Lifecycle strategy that toggles personalization tests for edge clients."""

    def process_server_response(self, context, server_response):
        if "personalization_test" in server_response:
            owner = context.owner
            if owner is not None:
                owner.do_personalization_test = True
            return

        super().process_server_response(context, server_response)


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""

    def __init__(self, server):
        super().__init__(server)
        self.do_personalization_test = False

        payload_strategy = self.payload_strategy
        training_strategy = self.training_strategy
        reporting_strategy = self.reporting_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=CsMamlEdgeLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=CsMamlEdgeTrainingStrategy(),
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )


class CsMamlEdgeTrainingStrategy(EdgeTrainingStrategy):
    """Training strategy that adds personalization support for CS-MAML edge clients."""

    async def train(self, context: ClientContext):
        owner = context.owner
        if owner is None:
            raise RuntimeError("CS-MAML edge training requires an owning client.")

        if getattr(owner, "do_personalization_test", False):
            owner.do_personalization_test = False
            return await self._run_personalization(owner)

        report, payload = await super().train(context)
        logging.info("[%s] Model aggregated on edge server.", owner)
        return report, payload

    async def _run_personalization(self, owner):
        logging.info(
            "[Edge Server #%d] Passing the global meta model to its clients.",
            owner.client_id,
        )

        await owner.server.select_testing_clients()
        await owner.server.per_accuracy_aggregated.wait()
        owner.server.per_accuracy_aggregated.clear()

        report = owner.server.personalization_accuracy
        payload = "personalization_accuracy"
        return report, payload
