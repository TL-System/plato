"""
A federated learning client using pruning.
"""

import copy
import logging

from fedsaw_algorithm import Algorithm as FedSawAlgorithm

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultTrainingStrategy
from plato.config import Config


class FedSawClientLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that records pruning amounts for FedSaw clients."""

    _STATE_KEY = "fedsaw_client"

    @staticmethod
    def _state(context):
        return context.state.setdefault(FedSawClientLifecycleStrategy._STATE_KEY, {})

    def process_server_response(self, context, server_response):
        super().process_server_response(context, server_response)
        amount = server_response.get("pruning_amount")
        if amount is None:
            return

        state = self._state(context)
        state["pruning_amount"] = amount

        owner = context.owner
        if owner is not None:
            owner.pruning_amount = amount


class FedSawTrainingStrategy(DefaultTrainingStrategy):
    """Training strategy that prunes local updates before transmission."""

    async def train(self, context: ClientContext):
        algorithm = context.algorithm
        if algorithm is None:
            raise RuntimeError("Algorithm is required for FedSaw training.")

        previous_weights = copy.deepcopy(algorithm.extract_weights())
        report, new_weights = await super().train(context)

        weight_updates = self._prune_updates(context, previous_weights, new_weights)
        logging.info("[Client #%d] Pruned its weight updates.", context.client_id)

        return report, weight_updates

    def _prune_updates(self, context, previous_weights, new_weights):
        algorithm = context.algorithm
        updates = algorithm.compute_weight_updates(previous_weights, new_weights)

        pruning_method = (
            "random"
            if getattr(Config().clients, "pruning_method", None) == "random"
            else "l1"
        )
        pruning_amount = getattr(context.owner, "pruning_amount", None)
        if pruning_amount is None:
            state = FedSawClientLifecycleStrategy._state(context)
            pruning_amount = state.get("pruning_amount", 0)

        return algorithm.prune_weight_updates(
            updates, amount=pruning_amount, method=pruning_method
        )


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks=None,
):
    """Build a FedSaw client that prunes its updates before reporting."""
    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm or FedSawAlgorithm,
        trainer=trainer,
        callbacks=callbacks,
        trainer_callbacks=trainer_callbacks,
    )
    client.pruning_amount = 0

    client._configure_composable(
        lifecycle_strategy=FedSawClientLifecycleStrategy(),
        payload_strategy=client.payload_strategy,
        training_strategy=FedSawTrainingStrategy(),
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


# Maintain compatibility for imports expecting a Client callable/class.
Client = create_client
