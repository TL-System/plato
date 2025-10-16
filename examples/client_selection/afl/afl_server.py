"""
A federated learning server using Active Federated Learning, where in each round
clients are selected not uniformly at random, but with a probability conditioned
on the current model, as well as the data on the client, to maximize efficiency.

Reference:

Goetz et al., "Active Federated Learning", 2019.

https://arxiv.org/pdf/1909.12641.pdf
"""

import logging
import math
import random

import numpy as np

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the AFL algorithm."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        self.local_values = {}

    def weights_aggregated(self, updates):
        """Extract required information from client reports after aggregating weights."""
        for update in updates:
            self.local_values[update.client_id]["valuation"] = update.report.valuation

    def calc_sample_distribution(self, clients_pool):
        """Calculate the sampling probability of each client for the next round."""
        # First, initialize valuations and probabilities when new clients are connected
        for client_id in clients_pool:
            if client_id not in self.local_values:
                self.local_values[client_id] = {}
                self.local_values[client_id]["valuation"] = -float("inf")
                self.local_values[client_id]["prob"] = 0.0

        # For a proportion of clients with the smallest valuations, reset these valuations
        # to negative infinities
        num_smallest = int(Config().algorithm.alpha1 * len(clients_pool))
        smallest_valuations = dict(
            sorted(self.local_values.items(), key=lambda item: item[1]["valuation"])[
                :num_smallest
            ]
        )

        for client_id in smallest_valuations.keys():
            self.local_values[client_id]["valuation"] = -float("inf")

        for client_id in clients_pool:
            self.local_values[client_id]["prob"] = math.exp(
                Config().algorithm.alpha2 * self.local_values[client_id]["valuation"]
            )

    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        assert clients_count <= len(clients_pool)
        random.setstate(self.prng_state)
        # Update the clients sampling distribution
        self.calc_sample_distribution(clients_pool)

        # 1. Sample a subset of the clients according to the sampling distribution
        num1 = int(math.floor((1 - Config().algorithm.alpha3) * clients_count))
        weighted_candidates = [
            client_id
            for client_id in clients_pool
            if self.local_values[client_id]["prob"] > 0.0
        ]
        num1 = min(num1, len(weighted_candidates))

        subset1 = []
        if num1 > 0:
            probs = np.array(
                [
                    self.local_values[client_id]["prob"]
                    for client_id in weighted_candidates
                ]
            )
            total_prob = probs.sum()
            if total_prob <= 0:
                probs = np.ones(len(weighted_candidates), dtype=float) / len(
                    weighted_candidates
                )
            else:
                probs = probs / total_prob
            subset1 = np.random.choice(
                weighted_candidates, num1, p=probs, replace=False
            ).tolist()

        # 2. Sample a subset of the remaining clients uniformly at random
        num2 = clients_count - len(subset1)
        remaining = clients_pool.copy()

        for client_id in subset1:
            remaining.remove(client_id)

        subset2 = random.sample(remaining, num2) if num2 > 0 else []

        # 3. Selected clients are the union of these two subsets
        selected_clients = subset1 + subset2

        self.prng_state = random.getstate()
        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients
