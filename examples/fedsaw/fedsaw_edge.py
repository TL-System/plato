"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import copy
from collections import OrderedDict
import logging
import torch
from torch.nn.utils import prune

from plato.clients import edge
from plato.config import Config
from plato.models import registry as models_registry


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""
    async def train(self):
        """ The training process on a FedSaw edge client. """
        previous_weights = copy.deepcopy(
            self.server.algorithm.extract_weights())

        # Perform model training
        self.report, _ = await super().train()

        weight_updates = self.prune_updates(previous_weights)

        logging.info("[Edge Server #%d] Pruned its aggregated updates.",
                     self.client_id)

        return self.report, weight_updates

    def prune_updates(self, previous_weights):
        """ Prune aggregated updates. """

        updates = self.compute_weight_updates(previous_weights)
        updates_model = models_registry.get()
        updates_model.load_state_dict(updates, strict=True)

        parameters_to_prune = []
        for _, module in updates_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(
                    module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=Config().clients.pruning_amount if hasattr(
                Config().clients, 'pruning_amount') else 0.2,
        )

        for module, name in parameters_to_prune:
            prune.remove(module, name)

        return updates_model.cpu().state_dict()

    def compute_weight_updates(self, previous_weights):
        """ Compute the weight updates. """
        # Extract trained model weights
        new_weights = self.server.algorithm.extract_weights()

        # Calculate updates from the received weights
        updates = OrderedDict()
        for name, new_weight in new_weights.items():
            previous_weight = previous_weights[name]

            # Calculate update
            delta = new_weight - previous_weight
            updates[name] = delta

        return updates
