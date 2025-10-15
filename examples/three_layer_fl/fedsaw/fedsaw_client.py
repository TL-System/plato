"""
A federated learning client using pruning.
"""

import copy
import logging
from collections import OrderedDict

import torch
from torch.nn.utils import prune

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
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


class Client(simple.Client):
    """
    A federated learning client prunes its update before sending out.
    """

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(
            model=model, datasource=datasource, algorithm=algorithm, trainer=trainer
        )
        self.pruning_amount = 0

        payload_strategy = self.payload_strategy
        training_strategy = self.training_strategy
        reporting_strategy = self.reporting_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=FedSawClientLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )

    async def _train(self):
        """The training process on a FedSaw client."""
        previous_weights = copy.deepcopy(self.algorithm.extract_weights())

        # Perform model training
        self._report, new_weights = await super()._train()

        weight_updates = self.prune_updates(previous_weights, new_weights)
        logging.info("[Client #%d] Pruned its weight updates.", self.client_id)

        return self._report, weight_updates

    def prune_updates(self, previous_weights, new_weights):
        """Prunes locally trained updates."""
        updates = self.compute_weight_updates(previous_weights, new_weights)
        self.algorithm.load_weights(updates)
        updates_model = self.algorithm.model

        parameters_to_prune = []
        for _, module in updates_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(
                module, torch.nn.Linear
            ):
                parameters_to_prune.append((module, "weight"))

        if (
            hasattr(Config().clients, "pruning_method")
            and Config().clients.pruning_method == "random"
        ):
            pruning_method = prune.RandomUnstructured
        else:
            pruning_method = prune.L1Unstructured

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=self.pruning_amount,
        )

        for module, name in parameters_to_prune:
            prune.remove(module, name)

        return updates_model.cpu().state_dict()

    def compute_weight_updates(self, previous_weights, new_weights):
        """Compute the weight updates."""
        # Calculate deltas from the received weights
        deltas = OrderedDict()
        for name, new_weight in new_weights.items():
            previous_weight = previous_weights[name]

            # Calculate deltas
            delta = new_weight - previous_weight
            deltas[name] = delta

        return deltas
