"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import copy
import logging

from fedsaw_algorithm import Algorithm as FedSawAlgorithm

from plato.clients import edge
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.edge import EdgeLifecycleStrategy, EdgeTrainingStrategy
from plato.config import Config


class FedSawEdgeLifecycleStrategy(EdgeLifecycleStrategy):
    """Lifecycle strategy that records pruning amounts for FedSaw edge clients."""

    def process_server_response(self, context, server_response):
        super().process_server_response(context, server_response)

        pruning_amounts = server_response.get("pruning_amount")
        if pruning_amounts is None:
            return

        logical_client_id = Config().args.id
        try:
            pruning_amount = pruning_amounts[str(logical_client_id)]
        except (KeyError, TypeError):
            return

        owner = context.owner
        if owner is not None and hasattr(owner, "server"):
            owner.server.edge_pruning_amount = pruning_amount


class FedSawEdgeTrainingStrategy(EdgeTrainingStrategy):
    """Training strategy that prunes aggregated updates on the edge server."""

    async def train(self, context: ClientContext):
        owner = context.owner
        if owner is None:
            raise RuntimeError("FedSaw edge strategy requires an owning client.")

        server = owner.server
        algorithm = server.algorithm
        previous_weights = copy.deepcopy(algorithm.extract_weights())

        report, new_weights = await super().train(context)

        weight_updates = self._prune_updates(context, previous_weights, new_weights)
        logging.info(
            "[Edge Server #%d] Pruned its aggregated updates.", context.client_id
        )

        return report, weight_updates

    def _prune_updates(self, context, previous_weights, new_weights):
        owner = context.owner
        server = owner.server
        algorithm = server.algorithm

        updates = algorithm.compute_weight_updates(previous_weights, new_weights)

        pruning_method = (
            "random"
            if getattr(Config().clients, "pruning_method", None) == "random"
            else "l1"
        )

        return algorithm.prune_weight_updates(
            updates, amount=server.edge_pruning_amount, method=pruning_method
        )


def create_client(
    *,
    server,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
):
    """Build a FedSaw edge client with pruning-aware strategies."""
    client = edge.Client(
        server=server,
        model=model,
        datasource=datasource,
        algorithm=algorithm or FedSawAlgorithm,
        trainer=trainer,
        callbacks=callbacks,
    )

    client._configure_composable(
        lifecycle_strategy=FedSawEdgeLifecycleStrategy(),
        payload_strategy=client.payload_strategy,
        training_strategy=FedSawEdgeTrainingStrategy(),
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
