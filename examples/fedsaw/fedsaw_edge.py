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
from plato.models import registry as models_registry


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""

    async def train(self):
        """The training process on a FedSaw edge client."""
        previous_weights = copy.deepcopy(self.server.algorithm.extract_weights())

        self._report, _ = await super().train()

        weight_updates = self.prune_updates(previous_weights)
        logging.info("[Edge Server #%d] Pruned its aggregated updates.", self.client_id)

        return self._report, weight_updates

    def prune_updates(self, previous_weights):
        """Prune aggregated updates."""
        deltas = self.compute_weight_deltas(previous_weights)
        updates_model = models_registry.get()
        updates_model.load_state_dict(deltas, strict=True)

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
            amount=Config().clients.pruning_amount,
        )

        for module, name in parameters_to_prune:
            prune.remove(module, name)

        return updates_model.cpu().state_dict()

    def compute_weight_deltas(self, previous_weights):
        """Compute the weight deltas."""
        # Extract trained model weights
        new_weights = self.server.algorithm.extract_weights()

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

        if "pruning_amount" in server_response:
            pruning_amount_list = server_response["pruning_amount"]
            pruning_amount = pruning_amount_list[
                self.client_id - Config().clients.total_clients - 1
            ]
            # Update pruning amount
            Config().clients = Config().clients._replace(pruning_amount=pruning_amount)
