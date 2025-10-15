"""
Active Federated Learning (AFL) client selection strategy.
"""

from __future__ import annotations

import logging
import math
import random
from types import SimpleNamespace
from typing import List

import numpy as np

from plato.config import Config
from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext


class AFLSelectionStrategy(ClientSelectionStrategy):
    """Select clients based on valuation metrics to maximise global utility."""

    def __init__(self, alpha1: float = 0.75, alpha2: float = 0.01, alpha3: float = 0.1):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.local_values: dict[int, dict[str, float]] = {}

    def setup(self, context: ServerContext) -> None:
        """Load parameters from configuration if available."""
        try:
            algo_cfg = Config().algorithm
            if hasattr(algo_cfg, "alpha1"):
                self.alpha1 = algo_cfg.alpha1
            if hasattr(algo_cfg, "alpha2"):
                self.alpha2 = algo_cfg.alpha2
            if hasattr(algo_cfg, "alpha3"):
                self.alpha3 = algo_cfg.alpha3
        except ValueError:
            pass

        logging.info(
            "AFL: alpha1=%.2f alpha2=%.3f alpha3=%.2f",
            self.alpha1,
            self.alpha2,
            self.alpha3,
        )

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        """Select clients using AFL's valuation-based policy."""
        assert clients_count <= len(clients_pool)

        for client_id in clients_pool:
            self.local_values.setdefault(
                client_id, {"valuation": -float("inf"), "prob": 0.0}
            )

        self._calc_sample_distribution(clients_pool)

        prng_state = context.state.get("prng_state")
        if prng_state:
            random.setstate(prng_state)

        num_weighted = int(math.floor((1 - self.alpha3) * clients_count))
        probs = np.array([self.local_values[cid]["prob"] for cid in clients_pool])
        probs = probs + 0.01
        probs /= probs.sum()

        subset_weighted = np.random.choice(
            clients_pool, num_weighted, p=probs, replace=False
        ).tolist()

        remaining = [c for c in clients_pool if c not in subset_weighted]
        subset_random = random.sample(remaining, clients_count - num_weighted)

        selected_clients = subset_weighted + subset_random
        context.state["prng_state"] = random.getstate()

        logging.info("[Server] AFL selected clients: %s", selected_clients)
        return selected_clients

    def on_reports_received(
        self, updates: List[SimpleNamespace], context: ServerContext
    ) -> None:
        """Update stored valuations from client reports."""
        for update in updates:
            if hasattr(update.report, "valuation"):
                self.local_values.setdefault(update.client_id, {}).setdefault(
                    "prob", 0.0
                )
                self.local_values[update.client_id]["valuation"] = (
                    update.report.valuation
                )
                logging.debug(
                    "AFL: Client #%d valuation = %.4f",
                    update.client_id,
                    update.report.valuation,
                )

    def _calc_sample_distribution(self, clients_pool: List[int]) -> None:
        """Calculate sampling probabilities for the current pool."""
        num_smallest = int(self.alpha1 * len(clients_pool))
        sorted_clients = sorted(
            self.local_values.items(), key=lambda item: item[1]["valuation"]
        )[:num_smallest]

        for client_id, _ in sorted_clients:
            self.local_values[client_id]["valuation"] = -float("inf")

        for client_id in clients_pool:
            valuation = self.local_values[client_id]["valuation"]
            self.local_values[client_id]["prob"] = (
                0.0 if valuation == -float("inf") else math.exp(self.alpha2 * valuation)
            )

        total_prob = sum(self.local_values[cid]["prob"] for cid in clients_pool)
        if total_prob == 0:
            uniform_prob = 1.0 / len(clients_pool)
            for client_id in clients_pool:
                self.local_values[client_id]["prob"] = uniform_prob
        else:
            for client_id in clients_pool:
                self.local_values[client_id]["prob"] /= total_prob
