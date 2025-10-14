"""
Client selection strategy implementations.

This module provides ready-to-use client selection strategies for various
federated learning scenarios.
"""

import logging
import math
import random
from types import SimpleNamespace
from typing import List

import numpy as np

from plato.config import Config
from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext


class RandomSelectionStrategy(ClientSelectionStrategy):
    """
    Random client selection (uniform sampling).

    Selects clients uniformly at random from the pool. This is the default
    and most common selection strategy in federated learning.

    Example:
        >>> strategy = RandomSelectionStrategy()
        >>> server = fedavg.Server(client_selection_strategy=strategy)
    """

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        """Select clients uniformly at random."""
        assert clients_count <= len(clients_pool), (
            f"Cannot select {clients_count} clients from pool of {len(clients_pool)}"
        )

        # Use server's PRNG state for reproducibility
        prng_state = context.state.get("prng_state")
        if prng_state:
            random.setstate(prng_state)

        selected_clients = random.sample(clients_pool, clients_count)

        # Save PRNG state back to context
        context.state["prng_state"] = random.getstate()

        logging.info("[Server] Selected clients: %s", selected_clients)
        return selected_clients


class OortSelectionStrategy(ClientSelectionStrategy):
    """
    Oort utility-based client selection.

    Combines exploration and exploitation using client utilities based on
    statistical utility, training times, and staleness. Maintains a blacklist
    of frequently selected clients.

    Reference:
        Lai et al., "Oort: Efficient Federated Learning via Guided
        Participant Selection", OSDI 2021.

    Args:
        exploration_factor: Proportion of clients for exploration (default: 0.3)
        desired_duration: Target round duration in seconds (default: 100.0)
        step_window: Window size for adjusting pacer (default: 10)
        penalty: Penalty exponent for slow clients (default: 0.8)
        cut_off: Utility cutoff multiplier (default: 0.95)
        blacklist_num: Number of selections before blacklisting (default: 10)

    Example:
        >>> strategy = OortSelectionStrategy(
        ...     exploration_factor=0.3,
        ...     desired_duration=100.0,
        ...     blacklist_num=15
        ... )
        >>> server = fedavg.Server(client_selection_strategy=strategy)
    """

    def __init__(
        self,
        exploration_factor: float = 0.3,
        desired_duration: float = 100.0,
        step_window: int = 10,
        penalty: float = 0.8,
        cut_off: float = 0.95,
        blacklist_num: int = 10,
    ):
        """Initialize Oort selection strategy."""
        super().__init__()
        self.exploration_factor = exploration_factor
        self.desired_duration = desired_duration
        self.step_window = step_window
        self.penalty = penalty
        self.cut_off = cut_off
        self.blacklist_num = blacklist_num

        # State maintained across rounds
        self.blacklist = []
        self.client_utilities = {}
        self.client_durations = {}
        self.client_last_rounds = {}
        self.client_selected_times = {}
        self.explored_clients = []
        self.unexplored_clients = []
        self.util_history = []
        self.pacer_step = desired_duration

    def setup(self, context: ServerContext) -> None:
        """Initialize client tracking dictionaries."""
        # Load from config if available
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
            # Config not initialized, use constructor parameters
            pass

        total_clients = context.total_clients

        # Initialize tracking dictionaries for all clients
        self.client_utilities = {
            client_id: 0 for client_id in range(1, total_clients + 1)
        }
        self.client_durations = {
            client_id: 0 for client_id in range(1, total_clients + 1)
        }
        self.client_last_rounds = {
            client_id: 0 for client_id in range(1, total_clients + 1)
        }
        self.client_selected_times = {
            client_id: 0 for client_id in range(1, total_clients + 1)
        }
        self.unexplored_clients = list(range(1, total_clients + 1))

        logging.info(
            "Oort: Initialized with exploration_factor=%.2f, desired_duration=%.1f, "
            "blacklist_num=%d",
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
        selected_clients = []
        current_round = context.current_round

        if current_round > 1:
            # Exploitation phase: select high-utility clients
            exploited_clients_count = max(
                math.ceil((1.0 - self.exploration_factor) * clients_count),
                clients_count - len(self.unexplored_clients),
            )

            # Sort clients by utility (descending)
            sorted_by_utility = sorted(
                self.client_utilities, key=self.client_utilities.get, reverse=True
            )
            sorted_by_utility = [
                client for client in sorted_by_utility if client in clients_pool
            ]

            # Calculate cut-off utility
            if len(sorted_by_utility) >= exploited_clients_count:
                cut_off_util = (
                    self.client_utilities[
                        sorted_by_utility[exploited_clients_count - 1]
                    ]
                    * self.cut_off
                )
            else:
                cut_off_util = 0

            # Include clients with utilities higher than the cut-off
            exploited_clients = [
                client_id
                for client_id in sorted_by_utility
                if (
                    self.client_utilities[client_id] > cut_off_util
                    and client_id not in self.blacklist
                )
            ]

            # Sample clients with their utilities as probabilities
            if exploited_clients:
                total_utility = float(
                    sum(
                        self.client_utilities[client_id]
                        for client_id in exploited_clients
                    )
                )

                if total_utility > 0:
                    probabilities = np.array(
                        [
                            self.client_utilities[client_id] / total_utility
                            for client_id in exploited_clients
                        ]
                    )
                    # Normalize to ensure sum is exactly 1.0
                    probabilities = probabilities / probabilities.sum()

                    num_to_select = min(len(exploited_clients), exploited_clients_count)
                    selected_clients = np.random.choice(
                        exploited_clients, num_to_select, p=probabilities, replace=False
                    ).tolist()

            # Fill remaining slots from sorted list if needed
            if len(selected_clients) < exploited_clients_count and exploited_clients:
                last_index = (
                    sorted_by_utility.index(exploited_clients[-1])
                    if exploited_clients
                    else 0
                )

                for index in range(last_index + 1, len(sorted_by_utility)):
                    if (
                        sorted_by_utility[index] not in self.blacklist
                        and len(selected_clients) < exploited_clients_count
                    ):
                        selected_clients.append(sorted_by_utility[index])

        # Exploration phase: select unexplored clients randomly
        prng_state = context.state.get("prng_state")
        if prng_state:
            random.setstate(prng_state)

        remaining_count = clients_count - len(selected_clients)
        if remaining_count > 0 and self.unexplored_clients:
            num_to_explore = min(remaining_count, len(self.unexplored_clients))
            selected_unexplore_clients = random.sample(
                self.unexplored_clients, num_to_explore
            )

            self.explored_clients += selected_unexplore_clients

            for client_id in selected_unexplore_clients:
                self.unexplored_clients.remove(client_id)

            selected_clients += selected_unexplore_clients

        context.state["prng_state"] = random.getstate()

        # Track selection counts
        for client in selected_clients:
            self.client_selected_times[client] += 1

        logging.info("[Server] Oort selected clients: %s", selected_clients)
        return selected_clients

    def on_reports_received(
        self, updates: List[SimpleNamespace], context: ServerContext
    ) -> None:
        """Update client utilities and durations after reports."""
        for update in updates:
            client_id = update.client_id

            # Extract statistical utility and training times
            if hasattr(update.report, "statistical_utility"):
                self.client_utilities[client_id] = update.report.statistical_utility
            if hasattr(update.report, "training_time"):
                self.client_durations[client_id] = update.report.training_time

            self.client_last_rounds[client_id] = context.current_round

            # Recalculate client utility with exploration bonus and penalty
            self.client_utilities[client_id] = self._calc_client_util(
                client_id, context.current_round
            )

        # Adjust pacer based on utility history
        if hasattr(update.report, "statistical_utility"):
            self.util_history.append(
                sum(update.report.statistical_utility for update in updates)
            )

            if context.current_round >= 2 * self.step_window:
                last_pacer_rounds = sum(
                    self.util_history[-2 * self.step_window : -self.step_window]
                )
                current_pacer_rounds = sum(self.util_history[-self.step_window :])
                if last_pacer_rounds > current_pacer_rounds:
                    self.desired_duration += self.pacer_step
                    logging.debug(
                        "Oort: Adjusted desired_duration to %.1f", self.desired_duration
                    )

        # Blacklist clients who have been selected too many times
        for update in updates:
            if self.client_selected_times[update.client_id] > self.blacklist_num:
                if update.client_id not in self.blacklist:
                    self.blacklist.append(update.client_id)
                    logging.info("Oort: Blacklisted client #%d", update.client_id)

    def _calc_client_util(self, client_id: int, current_round: int) -> float:
        """Calculate client utility with exploration bonus and duration penalty."""
        # Base utility with exploration bonus (UCB-style)
        last_round = max(1, self.client_last_rounds[client_id])
        exploration_bonus = math.sqrt(0.1 * math.log(current_round) / last_round)
        client_utility = self.client_utilities[client_id] + exploration_bonus

        # Apply duration penalty if client is too slow
        if (
            self.client_durations[client_id] > 0
            and self.desired_duration < self.client_durations[client_id]
        ):
            global_utility = (
                self.desired_duration / self.client_durations[client_id]
            ) ** self.penalty
            client_utility *= global_utility

        return client_utility


class AFLSelectionStrategy(ClientSelectionStrategy):
    """
    Active Federated Learning (AFL) client selection.

    Selects clients based on valuation, which measures how much a client
    can improve the global model. Combines value-based sampling with
    uniform random sampling.

    Reference:
        Goetz et al., "Active Federated Learning", 2019.

    Args:
        alpha1: Proportion of clients to reset valuations (default: 0.75)
        alpha2: Temperature parameter for sampling (default: 0.01)
        alpha3: Proportion for uniform random sampling (default: 0.1)

    Example:
        >>> strategy = AFLSelectionStrategy(
        ...     alpha1=0.75,
        ...     alpha2=0.01,
        ...     alpha3=0.1
        ... )
        >>> server = fedavg.Server(client_selection_strategy=strategy)
    """

    def __init__(self, alpha1: float = 0.75, alpha2: float = 0.01, alpha3: float = 0.1):
        """Initialize AFL selection strategy."""
        super().__init__()
        self.alpha1 = alpha1  # Proportion to reset valuations
        self.alpha2 = alpha2  # Temperature for sampling
        self.alpha3 = alpha3  # Proportion for uniform sampling
        self.local_values = {}

    def setup(self, context: ServerContext) -> None:
        """Load parameters from config if available."""
        try:
            if hasattr(Config().algorithm, "alpha1"):
                self.alpha1 = Config().algorithm.alpha1
            if hasattr(Config().algorithm, "alpha2"):
                self.alpha2 = Config().algorithm.alpha2
            if hasattr(Config().algorithm, "alpha3"):
                self.alpha3 = Config().algorithm.alpha3
        except ValueError:
            # Config not initialized, use constructor parameters
            pass

        logging.info(
            "AFL: Initialized with alpha1=%.2f, alpha2=%.3f, alpha3=%.2f",
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
        """Select clients using AFL algorithm."""
        assert clients_count <= len(clients_pool)

        # Initialize new clients with negative infinity valuation
        for client_id in clients_pool:
            if client_id not in self.local_values:
                self.local_values[client_id] = {
                    "valuation": -float("inf"),
                    "prob": 0.0,
                }

        # Update sampling distribution
        self._calc_sample_distribution(clients_pool)

        prng_state = context.state.get("prng_state")
        if prng_state:
            random.setstate(prng_state)

        # Phase 1: Sample based on valuations
        num1 = int(math.floor((1 - self.alpha3) * clients_count))
        probs = np.array([self.local_values[cid]["prob"] for cid in clients_pool])

        # Add small probability to zeros to avoid NaN
        probs = probs + 0.01
        probs /= probs.sum()

        subset1 = np.random.choice(clients_pool, num1, p=probs, replace=False).tolist()

        # Phase 2: Uniform random sampling from remaining clients
        num2 = clients_count - num1
        remaining = [c for c in clients_pool if c not in subset1]
        subset2 = random.sample(remaining, num2)

        selected_clients = subset1 + subset2

        context.state["prng_state"] = random.getstate()

        logging.info("[Server] AFL selected clients: %s", selected_clients)
        return selected_clients

    def on_reports_received(
        self, updates: List[SimpleNamespace], context: ServerContext
    ) -> None:
        """Extract valuations from client reports."""
        for update in updates:
            if hasattr(update.report, "valuation"):
                self.local_values[update.client_id]["valuation"] = (
                    update.report.valuation
                )
                logging.debug(
                    "AFL: Client #%d valuation = %.4f",
                    update.client_id,
                    update.report.valuation,
                )

    def _calc_sample_distribution(self, clients_pool: List[int]) -> None:
        """Calculate sampling probabilities for clients."""
        # Reset smallest valuations to negative infinity
        num_smallest = int(self.alpha1 * len(clients_pool))
        sorted_clients = sorted(
            self.local_values.items(), key=lambda x: x[1]["valuation"]
        )[:num_smallest]

        for client_id, _ in sorted_clients:
            self.local_values[client_id]["valuation"] = -float("inf")

        # Calculate probabilities using exponential weighting
        for client_id in clients_pool:
            valuation = self.local_values[client_id]["valuation"]
            # Avoid overflow with very large valuations
            if valuation == -float("inf"):
                self.local_values[client_id]["prob"] = 0.0
            else:
                self.local_values[client_id]["prob"] = math.exp(self.alpha2 * valuation)
