"""
A federated learning server using AFL.

Reference:

Goetz et al., "Active Federated Learning".

https://arxiv.org/pdf/1909.12641.pdf
"""
from collections import OrderedDict

from plato.servers import fedavg

import torch
import torch.nn.functional as F
import math
import random
from operator import itemgetter


class Server(fedavg.Server):
    """A federated learning server using the AFL algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.local_valuations = {}
        # Initialize the local valuations as negative infinities
        for client_id, client in dict(self.clients).items():
            self.local_valuations[client_id] = -float("inf")
        self.sample_probs = [None] * len(self.clients)
        # The proportion of clients with the smallest valuations to be set to negative infinities
        alpha1 = 0.75
        # The softmax temperature used in distribution
        alpha2 = 0.01
        # The proportion of clients which are selected uniformly at random
        alpha3 = 0.1

    async def federated_averaging(self, updates):
        """ Aggregate weight updates and deltas updates from the clients. """
        update = await super().federated_averaging(updates)

        # Extracting weights from the updates
        weights_received = self.extract_client_updates(updates)

        # Update the local valuations from the updates
        for i, update in enumerate(weights_received):
            report, __ = updates[i]
            client_id = self.selected_clients[i]
            self.local_valuations[client_id] = report.valuation
        
        return update

    def calc_sample_distribution(self):
        """Calculate the sampling probability of each client for the next round."""
        # For a proportion of clients with smallest valuations, reset these valuations to negative infinities
        num = int(math.ceil(self.alpha1 * len(self.clients)))
        smallest_valuations = dict(sorted(self.local_valuations.items, key=lambda item: item[1])[:num])
        # smallest_valuations = dict(sorted(self.local_valuations.items, key=itemgetter(1))[:num])
        for client_id in smallest_valuations.keys():
            self.local_valuations[client_id] = -float("inf")
        for k in range(len(self.clients)):
            self.sample_probs[k] = math.exp(self.alpha2 * self.local_valuations[k+1])

    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""
        # 1. Sample a subset of the clients according to the sampling distribution
        num1 = int(math.ceil((1 - self.alpha3) * self.clients_per_round))
        subset1 = random.choice(list(self.clients), weights=self.sample_probs, k=num1)
        # 2. Sample a subset of the remaining clients uniformly at random
        num2 = self.clients_per_round - num1
        subset2 = random.sample(list(self.clients) - subset1, num2)
        # 3. Chosed clients are the union of these two subsets
        return subset1 + subset2
