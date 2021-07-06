"""
A federated learning server using AFL.

Reference:

Goetz et al., "Active Federated Learning".

https://arxiv.org/pdf/1909.12641.pdf
"""
from collections import OrderedDict

from plato.servers import fedavg

import logging
import numpy as np
import math
import random


class Server(fedavg.Server):
    """A federated learning server using the AFL algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        # The proportion of clients with the smallest valuations to be set to negative infinities
        self.alpha1 = 0.75
        # The softmax temperature used in distribution
        self.alpha2 = 0.01
        # The proportion of clients which are selected uniformly at random
        self.alpha3 = 0.1

    async def federated_averaging(self, updates):
        """ Aggregate weight updates and deltas updates from the clients. """
        update = await super().federated_averaging(updates)

        # Extracting weights from the updates
        weights_received = self.extract_client_updates(updates)
        
        # Update the local valuations from the updates
        for i, update in enumerate(weights_received):
            report, __ = updates[i]
            client_id = self.selected_clients[i]
            self.clients[client_id]["valuation"] = report.valuation
        
        return update

    def calc_sample_distribution(self):
        """Calculate the sampling probability of each client for the next round."""
        # Initialize valuations and probabilities when new clients are connected
        for client_id, client in self.clients.items():
            if "valuation" not in client:
                client["valuation"] = -float("inf")
            if "prob" not in client:
                client["prob"] = 0.0

        # For a proportion of clients with smallest valuations, reset these valuations to negative infinities
        num_smallest = int(self.alpha1 * len(self.clients))
        smallest_valuations = dict(sorted(self.clients.items(), key=lambda item: item[1]["valuation"])[:num_smallest])
        for client_id in smallest_valuations.keys():
            self.clients[client_id]["valuation"] = -float("inf")
        for client_id, client in self.clients.items():
            client["prob"] = math.exp(self.alpha2 * client["valuation"])

    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""
        # Update the clients sampling distribution
        self.calc_sample_distribution()
        # 1. Sample a subset of the clients according to the sampling distribution
        num1 = int(math.floor((1 - self.alpha3) * self.clients_per_round))
        pool = list(self.clients)
        probs = np.array([self.clients[client_id]["prob"] for client_id in pool])
        if probs.sum() != 0.0:
            probs /= probs.sum()
        else:
            probs = None
        subset1 = np.random.choice(pool, num1, p=probs,replace=False).tolist()
        for i in subset1:
            logging.info("client in subset1: %s", i)
        # 2. Sample a subset of the remaining clients uniformly at random
        num2 = self.clients_per_round - num1
        remaining = pool
        for client_id in subset1:
            remaining.remove(client_id)
        for i in remaining:
            logging.info("client in remaining: %s", i)
        subset2 = random.sample(remaining, num2)
        for i in subset2:
            logging.info("client in subset2: %s", i)
        # 3. Selected clients are the union of these two subsets
        return subset1 + subset2
