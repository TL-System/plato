"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import copy
import logging
from collections import OrderedDict

import torch
from torch.nn.utils import prune

from plato.clients import edge
from plato.clients.strategies.edge import EdgeLifecycleStrategy
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


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""

    async def _train(self):
        """The training process on a FedSaw edge client."""
        previous_weights = copy.deepcopy(self.server.algorithm.extract_weights())

        self._report, new_weights = await super()._train()

        weight_updates = self.prune_updates(previous_weights, new_weights)
        logging.info("[Edge Server #%d] Pruned its aggregated updates.", self.client_id)

        return self._report, weight_updates

    def prune_updates(self, previous_weights, new_weights):
        """Prunes aggregated updates."""
        updates = self.compute_weight_updates(previous_weights, new_weights)
        self.server.algorithm.load_weights(updates)
        updates_model = self.server.algorithm.model

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
            amount=self.server.edge_pruning_amount,
        )

        for module, name in parameters_to_prune:
            prune.remove(module, name)

        return updates_model.cpu().state_dict()

    def __init__(
        self,
        server,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            server=server,
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        payload_strategy = self.payload_strategy
        training_strategy = self.training_strategy
        reporting_strategy = self.reporting_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=FedSawEdgeLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )

    def compute_weight_updates(self, previous_weights, new_weights):
        """Computes the weight updates."""
        # Calculate deltas from the received weights
        deltas = OrderedDict()
        for name, new_weight in new_weights.items():
            previous_weight = previous_weights[name]

            # Calculate deltas
            delta = new_weight - previous_weight
            deltas[name] = delta

        return deltas
