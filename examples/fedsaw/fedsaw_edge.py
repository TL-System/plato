"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import copy
from collections import OrderedDict
from dataclasses import dataclass
import logging
import time
import torch
from torch.nn.utils import prune

from plato.clients import edge
from plato.config import Config
from plato.models import registry as models_registry


@dataclass
class Report(edge.Report):
    """ Client report, to be sent to the federated learning server. """


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""

    async def train(self):
        """ The training process on a FedSaw edge client. """
        previous_weights = copy.deepcopy(
            self.server.algorithm.extract_weights())

        training_start_time = time.perf_counter()
        # Signal edge server to select clients to start a new round of local aggregation
        self.server.new_global_round_begins.set()

        # Wait for the edge server to finish model aggregation
        await self.server.model_aggregated.wait()
        self.server.model_aggregated.clear()

        average_accuracy = self.server.average_accuracy
        accuracy = self.server.accuracy

        training_time = time.perf_counter() - training_start_time

        comm_time = time.time()

        # Generate a report for the central server
        self.report = Report(self.server.total_samples, accuracy,
                             training_time, comm_time, False, average_accuracy,
                             self.client_id, self.server.comm_overhead)

        self.server.comm_overhead = 0
        weight_updates = self.prune_updates(previous_weights)

        logging.info("[Edge Server #%d] Pruned its aggregated updates.",
                     self.client_id)

        return self.report, weight_updates

    def prune_updates(self, previous_weights):
        """ Prune aggregated updates. """

        deltas = self.compute_weight_deltas(previous_weights)
        updates_model = models_registry.get()
        updates_model.load_state_dict(deltas, strict=True)

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
        """ Additional client-specific processing on the server response. """
        super().process_server_response(server_response)

        if 'pruning_amount' in server_response:
            pruning_amount_list = server_response['pruning_amount']
            if hasattr(Config().clients,
                       'simulation') and Config().clients.simulation:
                index = self.client_id - Config().clients.per_round - 1
            else:
                index = self.client_id - Config().clients.total_clients - 1

            pruning_amount = pruning_amount_list[index]
            # Update pruning amount
            Config().clients = Config().clients._replace(
                pruning_amount=pruning_amount)
