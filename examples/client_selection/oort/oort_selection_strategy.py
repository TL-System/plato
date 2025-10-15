"""
Oort client selection strategy.
"""

from __future__ import annotations

import logging
import math
import random
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

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

        # Runtime state initialised in setup()
        self.blacklist: List[int] = []
        self.client_utilities: Dict[int, float] = {}
        self.client_durations: Dict[int, float] = {}
        self.client_last_rounds: Dict[int, int] = {}
        self.client_selected_times: Dict[int, int] = {}
        self.explored_clients: List[int] = []
        self.unexplored_clients: List[int] = []
        self.util_history: List[float] = []
        self.pacer_step = desired_duration

    def setup(self, context: ServerContext) -> None:
        """Initialise tracking dictionaries using configuration values."""
        server_config = getattr(Config(), "server", None)
        if server_config is not None:
            self.exploration_factor = getattr(
                server_config, "exploration_factor", self.exploration_factor
            )
            self.desired_duration = getattr(
                server_config, "desired_duration", self.desired_duration
            )
            self.step_window = getattr(server_config, "step_window", self.step_window)
            self.penalty = getattr(server_config, "penalty", self.penalty)
            self.cut_off = getattr(server_config, "cut_off", self.cut_off)
            self.blacklist_num = getattr(
                server_config, "blacklist_num", self.blacklist_num
            )

        total_clients = context.total_clients

        self.blacklist = []
        self.client_utilities = {
            client_id: 0.0 for client_id in range(1, total_clients + 1)
        }
        self.client_durations = {
            client_id: 0.0 for client_id in range(1, total_clients + 1)
        }
        self.client_last_rounds = {
            client_id: 0 for client_id in range(1, total_clients + 1)
        }
        self.client_selected_times = {
            client_id: 0 for client_id in range(1, total_clients + 1)
        }

        self.explored_clients = []
        self.unexplored_clients = list(range(1, total_clients + 1))
        self.util_history = []
        self.pacer_step = self.desired_duration

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
        """Select clients using the Oort algorithm."""
        assert clients_count <= len(clients_pool)

        selected_clients: List[int] = []
        current_round = context.current_round

        prng_state = context.state.get("prng_state")
        if prng_state is not None:
            random.setstate(prng_state)

        if current_round > 1:
            exploited_clients_count = max(
                math.ceil((1.0 - self.exploration_factor) * clients_count),
                clients_count - len(self.unexplored_clients),
            )

            if exploited_clients_count > 0:
                sorted_by_utility = sorted(
                    self.client_utilities,
                    key=self.client_utilities.get,
                    reverse=True,
                )
                sorted_by_utility = [
                    client for client in sorted_by_utility if client in clients_pool
                ]

                if sorted_by_utility and exploited_clients_count <= len(
                    sorted_by_utility
                ):
                    cut_off_util = (
                        self.client_utilities[
                            sorted_by_utility[exploited_clients_count - 1]
                        ]
                        * self.cut_off
                    )
                else:
                    cut_off_util = 0.0

                exploited_candidates = [
                    client_id
                    for client_id in sorted_by_utility
                    if (
                        self.client_utilities[client_id] > cut_off_util
                        and client_id not in self.blacklist
                    )
                ]

                total_utility = float(
                    sum(
                        self.client_utilities[client_id]
                        for client_id in exploited_candidates
                    )
                )
                if (
                    exploited_candidates
                    and total_utility > 0.0
                    and exploited_clients_count > 0
                ):
                    probabilities = np.array(
                        [
                            self.client_utilities[client_id] / total_utility
                            for client_id in exploited_candidates
                        ]
                    )
                    probabilities = probabilities / probabilities.sum()

                    exploited_sample = np.random.choice(
                        exploited_candidates,
                        min(len(exploited_candidates), exploited_clients_count),
                        p=probabilities,
                        replace=False,
                    )
                    selected_clients.extend(exploited_sample.tolist())

                if selected_clients:
                    last_selected = selected_clients[-1]
                    last_index = (
                        sorted_by_utility.index(last_selected)
                        if last_selected in sorted_by_utility
                        else -1
                    )
                else:
                    last_index = -1

                while len(selected_clients) < exploited_clients_count:
                    last_index += 1
                    if last_index >= len(sorted_by_utility):
                        break
                    candidate = sorted_by_utility[last_index]
                    if candidate in self.blacklist:
                        continue
                    if candidate not in selected_clients:
                        selected_clients.append(candidate)

        remaining_slots = clients_count - len(selected_clients)
        if remaining_slots > 0:
            available_unexplored = [
                client
                for client in self.unexplored_clients
                if client in clients_pool and client not in selected_clients
            ]
            exploration_count = min(remaining_slots, len(available_unexplored))

            if exploration_count > 0:
                selected_unexplored = random.sample(
                    available_unexplored, exploration_count
                )
                self.explored_clients.extend(selected_unexplored)
                for client_id in selected_unexplored:
                    if client_id in self.unexplored_clients:
                        self.unexplored_clients.remove(client_id)
                selected_clients.extend(selected_unexplored)

        if len(selected_clients) < clients_count:
            remaining_candidates = [
                client
                for client in clients_pool
                if client not in selected_clients and client not in self.blacklist
            ]
            additional_needed = min(
                clients_count - len(selected_clients), len(remaining_candidates)
            )
            if additional_needed > 0:
                selected_clients.extend(
                    random.sample(remaining_candidates, additional_needed)
                )

        for client in selected_clients:
            self.client_selected_times[client] += 1

        context.state["prng_state"] = random.getstate()

        server = context.server if context.server is not None else "Server"
        logging.info("[%s] Oort selected clients: %s", server, selected_clients)
        return selected_clients

    def on_reports_received(
        self, updates: List[SimpleNamespace], context: ServerContext
    ) -> None:
        """Update utility statistics after each round."""
        current_round = context.current_round

        for update in updates:
            client_id = update.client_id
            report = update.report
            statistical_utility = getattr(report, "statistical_utility", 0.0)
            training_time = getattr(report, "training_time", 0.0)

            self.client_utilities[client_id] = statistical_utility
            self.client_durations[client_id] = training_time
            self.client_last_rounds[client_id] = max(current_round, 1)
            self.client_utilities[client_id] = self.calc_client_util(
                client_id, current_round
            )

        if updates:
            total_utility = sum(
                getattr(update.report, "statistical_utility", 0.0) for update in updates
            )
            self.util_history.append(total_utility)

        if self.step_window > 0 and len(self.util_history) >= 2 * self.step_window:
            last_window = sum(
                self.util_history[-2 * self.step_window : -self.step_window]
            )
            current_window = sum(self.util_history[-self.step_window :])
            if last_window > current_window:
                self.desired_duration += self.pacer_step

        for update in updates:
            client_id = update.client_id
            if (
                self.client_selected_times.get(client_id, 0) > self.blacklist_num
                and client_id not in self.blacklist
            ):
                self.blacklist.append(client_id)

    def calc_client_util(self, client_id: int, current_round: int) -> float:
        """Calculate the client utility with exploration-exploitation balance."""
        base_utility = self.client_utilities.get(client_id, 0.0)

        last_round = self.client_last_rounds.get(client_id, 1)
        exploration_bonus = 0.0
        if last_round > 0 and current_round > 1:
            exploration_bonus = math.sqrt(
                max(0.0, 0.1 * math.log(current_round) / last_round)
            )

        client_utility = base_utility + exploration_bonus

        client_duration = self.client_durations.get(client_id, 0.0)
        if client_duration > 0 and self.desired_duration < client_duration:
            global_utility = (self.desired_duration / client_duration) ** self.penalty
            client_utility *= global_utility

        return client_utility
