"""
A federated learning client using pruning.
"""

import copy
from collections import OrderedDict
import logging
import torch
from torch.nn.utils import prune

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """
    A federated learning client prunes its update before sending out.
    """

    async def train(self):
        """ The training process on a FedSaw client. """
        previous_weights = copy.deepcopy(self.algorithm.extract_weights())

        # Perform model training
        self.report, _ = await super().train()

        weight_updates = self.prune_updates(previous_weights)
        logging.info("[Client #%d] Pruned its weight updates.", self.client_id)

        return self.report, weight_updates

    def prune_updates(self, previous_weights):
        """ Prune locally trained updates. """

        updates = self.compute_weight_deltas(previous_weights)
        self.algorithm.load_weights(updates)
        updates_model = self.algorithm.model

        parameters_to_prune = []
        for _, module in updates_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(
                    module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))

        if hasattr(Config().clients, 'pruning_method') and Config(
        ).clients.pruning_method == 'random':
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
        """ Compute the weight deltas. """
        # Extract trained model weights
        new_weights = self.algorithm.extract_weights()

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
        if 'pruning_amount' in server_response:
            # Update pruning amount
            Config().clients = Config().clients._replace(
                pruning_amount=server_response['pruning_amount'])
