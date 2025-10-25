"""
A federated learning client using pruning.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any

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
    def _state(context: ClientContext) -> dict[str, Any]:
        return context.state.setdefault(FedSawClientLifecycleStrategy._STATE_KEY, {})

    def process_server_response(
        self, context: ClientContext, server_response: dict[str, Any]
    ) -> None:
        super().process_server_response(context, server_response)
        amount = server_response.get("pruning_amount")
        if amount is None:
            return

        state = self._state(context)
        state["pruning_amount"] = amount

        owner = context.owner
        if isinstance(owner, FedSawClient) and isinstance(amount, (int, float)):
            owner.pruning_amount = float(amount)


class FedSawTrainingStrategy(DefaultTrainingStrategy):
    """Training strategy that prunes local updates before transmission."""

    async def train(self, context: ClientContext):
        algorithm = context.algorithm
        if not isinstance(algorithm, FedSawAlgorithm):
            raise RuntimeError("FedSaw training requires a FedSaw algorithm instance.")

        previous_weights = copy.deepcopy(algorithm.extract_weights())
        report, new_weights = await super().train(context)

        weight_updates = self._prune_updates(context, previous_weights, new_weights)
        logging.info("[Client #%d] Pruned its weight updates.", context.client_id)

        return report, weight_updates

    def _prune_updates(
        self,
        context: ClientContext,
        previous_weights: Mapping[str, Any],
        new_weights: Mapping[str, Any],
    ):
        algorithm = context.algorithm
        if not isinstance(algorithm, FedSawAlgorithm):
            raise RuntimeError("FedSaw algorithm required to prune weight updates.")

        updates = algorithm.compute_weight_updates(previous_weights, new_weights)

        pruning_method = (
            "random"
            if getattr(Config().clients, "pruning_method", None) == "random"
            else "l1"
        )
        owner = context.owner
        pruning_amount: float | int | None = None
        if isinstance(owner, FedSawClient):
            pruning_amount = owner.pruning_amount

        if pruning_amount is None:
            state = FedSawClientLifecycleStrategy._state(context)
            stored_amount = state.get("pruning_amount", 0)
            pruning_amount = stored_amount if isinstance(stored_amount, (int, float)) else 0

        return algorithm.prune_weight_updates(
            updates, amount=pruning_amount, method=pruning_method
        )


class FedSawClient(simple.Client):
    """Client implementation that tracks pruning metadata for FedSaw."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm or FedSawAlgorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=trainer_callbacks,
        )
        self.pruning_amount: float = 0.0

        self._configure_composable(
            lifecycle_strategy=FedSawClientLifecycleStrategy(),
            payload_strategy=self.payload_strategy,
            training_strategy=FedSawTrainingStrategy(),
            reporting_strategy=self.reporting_strategy,
            communication_strategy=self.communication_strategy,
        )


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks=None,
) -> FedSawClient:
    """Build a FedSaw client that prunes its updates before reporting."""
    return FedSawClient(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
        trainer_callbacks=trainer_callbacks,
    )


# Maintain compatibility for imports expecting a Client callable/class.
Client = FedSawClient
