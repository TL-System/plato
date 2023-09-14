"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import copy
import logging
from collections import OrderedDict

import torch
from torch.nn.utils import prune

from plato.clients import edge
from plato.config import Config


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

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        super().process_server_response(server_response)

        pruning_amount_list = server_response["pruning_amount"]
        pruning_amount = pruning_amount_list[str(self.client_id)]
        # Update pruning amount
        self.server.edge_pruning_amount = pruning_amount
