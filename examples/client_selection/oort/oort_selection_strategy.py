"""
Oort client selection strategy.
"""

from __future__ import annotations

import logging
import math
import random
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

try:
    import mosek  # type: ignore
    from cvxopt import log, matrix, solvers, sparse
except ImportError:  # pragma: no cover - optional dependency
    mosek = None  # type: ignore
    log = matrix = solvers = sparse = None  # type: ignore

from plato.config import Config
from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext


class OortSelectionStrategy(ClientSelectionStrategy):
    """Oort utility-based client selection with exploration and exploitation."""

    def __init__(
        self,
        exploration_factor: float = 0.3,
        desired_duration: float = 100.0,
        step_window: int = 10,
        penalty: float = 0.8,
        cut_off: float = 0.95,
        blacklist_num: int = 10,
    ):
        super().__init__()
        self.exploration_factor = exploration_factor
        self.desired_duration = desired_duration
        self.step_window = step_window
        self.penalty = penalty
        self.cut_off = cut_off
        self.blacklist_num = blacklist_num

        self.blacklist: List[int] = []
        self.client_utilities: dict[int, float] = {}
        self.client_durations: dict[int, float] = {}
        self.client_last_rounds: dict[int, int] = {}
        self.client_selected_times: dict[int, int] = {}
        self.explored_clients: List[int] = []
        self.unexplored_clients: List[int] = []
        self.util_history: List[float] = []
        self.pacer_step = desired_duration

    def setup(self, context: ServerContext) -> None:
        """Initialise tracking dictionaries."""
        try:
            if hasattr(Config().server, "exploration_factor"):
                self.exploration_factor = Config().server.exploration_factor
            if hasattr(Config().server, "desired_duration"):
                self.desired_duration = Config().server.desired_duration
            if hasattr(Config().server, "step_window"):
                self.step_window = Config().server.step_window
            if hasattr(Config().server, "penalty"):
                self.penalty = Config().server.penalty
            if hasattr(Config().server, "cut_off"):
                self.cut_off = Config().server.cut_off
            if hasattr(Config().server, "blacklist_num"):
                self.blacklist_num = Config().server.blacklist_num
        except ValueError:
            pass

        total_clients = context.total_clients
        self.client_utilities = {client_id: 0 for client_id in range(1, total_clients + 1)}
        self.client_durations = {client_id: 0 for client_id in range(1, total_clients + 1)}
        self.client_last_rounds = {client_id: 0 for client_id in range(1, total_clients + 1)}
        self.client_selected_times = {client_id: 0 for client_id in range(1, total_clients + 1)}
        self.unexplored_clients = list(range(1, total_clients + 1))

        logging.info(
            "Oort: exploration_factor=%.2f desired_duration=%.1f blacklist_num=%d",
            self.exploration_factor,
            self.desired_duration,
            self.blacklist_num,
        )

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        """Select clients using Oort algorithm."""
        selected_clients: List[int] = []
        current_round = context.current_round

        if current_round > 1:
            exploited_clients_count = max(
                math.ceil((1.0 - self.exploration_factor) * clients_count),
                clients_count - len(self.unexplored_clients),
            )

            sorted_by_utility = sorted(
                self.client_utilities, key=self.client_utilities.get, reverse=True
            )
            sorted_by_utility = [
                client for client in sorted_by_utility if client in clients_pool
            ]

            if len(sorted_by_utility) >= exploited_clients_count:
                cut_off_util = (
                    self.client_utilities[
                        sorted_by_utility[exploited_clients_count - 1]
                    ]
                    * self.cut_off
                )
            else:
                cut_off_util = 0.0

            exploited_clients = [
                client
                for client in sorted_by_utility
                if self.client_utilities[client] >= cut_off_util
            ][:exploited_clients_count]

            selected_clients.extend(exploited_clients)

        remaining_slots = clients_count - len(selected_clients)

        if remaining_slots > 0 and self.unexplored_clients:
            unexplored = [
                client
                for client in self.unexplored_clients
                if client in clients_pool and client not in selected_clients
            ]
            exploration_count = min(remaining_slots, len(unexplored))
            selected_clients.extend(random.sample(unexplored, exploration_count))

        if len(selected_clients) < clients_count:
            remaining = [
                client
                for client in clients_pool
                if client not in selected_clients
            ]
            selected_clients.extend(
                random.sample(remaining, clients_count - len(selected_clients))
            )

        for client in selected_clients:
            if client in self.unexplored_clients:
                self.unexplored_clients.remove(client)

        logging.info("[Server] Oort selected clients: %s", selected_clients)
        return selected_clients

    def update_client_utilities(
        self,
        updates: List[SimpleNamespace],
        context: ServerContext,
    ) -> None:
        """Update utility statistics after each round."""
        current_round = context.current_round
        for update in updates:
            client_id = update.client_id
            report = update.report
            duration = getattr(report, "training_time", 0.0)
            utility = getattr(report, "utility", 0.0)

            self.client_selected_times[client_id] += 1
            self.client_last_rounds[client_id] = current_round
            self.client_utilities[client_id] = utility
            self.client_durations[client_id] = duration

        if updates:
            avg_util = float(
                np.mean([self.client_utilities[update.client_id] for update in updates])
            )
            self.util_history.append(avg_util)

    def get_blacklist(self) -> List[int]:
        """Return clients that should be blacklisted."""
        self.blacklist = [
            client_id
            for client_id, count in self.client_selected_times.items()
            if count >= self.blacklist_num
        ]
        return self.blacklist

    def cluster_clients_by_duration(
        self, clients_pool: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Cluster clients into fast/slow groups using DBSCAN."""
        durations = np.array(
            [[self.client_durations.get(client_id, 0.0)] for client_id in clients_pool]
        )
        clustering = DBSCAN(eps=5.0, min_samples=1).fit(durations)

        fast_clients = [
            client
            for client, label in zip(clients_pool, clustering.labels_)
            if label == 0
        ]
        slow_clients = [
            client
            for client, label in zip(clients_pool, clustering.labels_)
            if label != 0
        ]
        return fast_clients, slow_clients
