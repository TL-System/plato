"""
Client selection strategy implementations.

This module provides ready-to-use client selection strategies for various
federated learning scenarios.
"""

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

        # Normalize probabilities
        total_prob = sum(self.local_values[cid]["prob"] for cid in clients_pool)
        if total_prob == 0:
            # Fall back to uniform distribution
            uniform_prob = 1.0 / len(clients_pool)
            for client_id in clients_pool:
                self.local_values[client_id]["prob"] = uniform_prob
        else:
            for client_id in clients_pool:
                self.local_values[client_id]["prob"] /= total_prob


class PiscesSelectionStrategy(ClientSelectionStrategy):
    """
    Pisces client selection with staleness-aware utilities and optional robustness.

    Combines exploration/exploitation with decaying exploration probability and
    supports outlier detection via DBSCAN on client utilities.

    Reference:
        Jiang et al., "Pisces: Efficient Federated Learning via Guided Asynchronous
        Training," SoCC 2022.
    """

    def __init__(
        self,
        exploration_factor: float = 0.3,
        exploration_decaying_factor: float = 0.99,
        min_explore_factor: float = 0.1,
        staleness_factor: float = 1.0,
        robustness: bool = False,
        augmented_factor: int = 5,
        threshold_factor: float = 1.0,
        reliability_credit_initial: int = 5,
        history_window: int = 5,
    ):
        super().__init__()
        self.exploration_factor = exploration_factor
        self.exploration_decaying_factor = exploration_decaying_factor
        self.min_explore_factor = min_explore_factor
        self.staleness_factor = staleness_factor
        self.robustness = robustness
        self.augmented_factor = augmented_factor
        self.threshold_factor = threshold_factor
        self.reliability_credit_initial = reliability_credit_initial
        self.history_window = history_window

        # Strategy state
        self.client_utilities = {}
        self.client_staleness = {}
        self.explored_clients = set()
        self.unexplored_clients = []
        self.reliability_credit_record = {}
        self.detected_corrupted_clients = []
        self.model_versions_clients_dict = {}
        self.per_round = 0

    def setup(self, context: ServerContext) -> None:
        """Initialize state dictionaries and allow config overrides."""
        try:
            server_cfg = Config().server
            if hasattr(server_cfg, "exploration_factor"):
                self.exploration_factor = server_cfg.exploration_factor
            if hasattr(server_cfg, "exploration_decaying_factor"):
                self.exploration_decaying_factor = (
                    server_cfg.exploration_decaying_factor
                )
            if hasattr(server_cfg, "min_explore_factor"):
                self.min_explore_factor = server_cfg.min_explore_factor
            if hasattr(server_cfg, "staleness_factor"):
                self.staleness_factor = server_cfg.staleness_factor
            if hasattr(server_cfg, "robustness"):
                self.robustness = server_cfg.robustness
            if hasattr(server_cfg, "augmented_factor"):
                self.augmented_factor = server_cfg.augmented_factor
            if hasattr(server_cfg, "threshold_factor"):
                self.threshold_factor = server_cfg.threshold_factor
            if hasattr(server_cfg, "reliability_credit_initial"):
                self.reliability_credit_initial = server_cfg.reliability_credit_initial
        except ValueError:
            # Config not initialized, keep constructor defaults
            pass

        total_clients = context.total_clients
        self.client_utilities = {
            client_id: 0.0 for client_id in range(1, total_clients + 1)
        }
        self.client_staleness = {
            client_id: [] for client_id in range(1, total_clients + 1)
        }
        self.unexplored_clients = list(range(1, total_clients + 1))
        self.explored_clients = set()
        self.reliability_credit_record = {
            client_id: self.reliability_credit_initial
            for client_id in range(1, total_clients + 1)
        }
        self.detected_corrupted_clients = []
        self.model_versions_clients_dict = {}
        self.per_round = context.clients_per_round

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        """Select clients using Pisces exploration/exploitation policy."""
        assert clients_count <= len(clients_pool), (
            f"Cannot select {clients_count} clients from pool of {len(clients_pool)}"
        )

        available_clients = list(clients_pool)
        if self.robustness and self.detected_corrupted_clients:
            outliers = [
                client_id
                for client_id in available_clients
                if client_id in self.detected_corrupted_clients
            ]
            if outliers:
                logging.info(
                    "PiscesSelection: filtering detected outliers %s", outliers
                )
            available_clients = [
                client_id
                for client_id in available_clients
                if client_id not in self.detected_corrupted_clients
            ]

        effective_count = min(clients_count, len(available_clients))
        if effective_count == 0:
            logging.warning("PiscesSelection: no available clients to select.")
            return []

        selected_clients: List[int] = []
        current_round = context.current_round

        if current_round > 1:
            unexplored_available = [
                client_id
                for client_id in self.unexplored_clients
                if client_id in available_clients
            ]
            explored_available = [
                client_id
                for client_id in self.explored_clients
                if client_id in available_clients
            ]

            explored_clients_count = min(
                len(unexplored_available),
                np.random.binomial(effective_count, self.exploration_factor, 1)[0],
            )

            self.exploration_factor = max(
                self.exploration_factor * self.exploration_decaying_factor,
                self.min_explore_factor,
            )

            exploited_clients_target = max(0, effective_count - explored_clients_count)
            exploited_clients_count = min(
                len(explored_available), exploited_clients_target
            )

            sorted_by_utility = sorted(
                self.client_utilities, key=self.client_utilities.get, reverse=True
            )
            sorted_by_utility = [
                client for client in sorted_by_utility if client in explored_available
            ]

            selected_clients = sorted_by_utility[:exploited_clients_count]

        prng_state = context.state.get("prng_state")
        if prng_state:
            random.setstate(prng_state)

        remaining_slots = effective_count - len(selected_clients)
        if remaining_slots > 0:
            exploration_candidates = [
                client_id
                for client_id in self.unexplored_clients
                if client_id in available_clients and client_id not in selected_clients
            ]
            explore_count = min(remaining_slots, len(exploration_candidates))

            if explore_count > 0:
                selected_unexplored = random.sample(
                    exploration_candidates, explore_count
                )
                self.explored_clients.update(selected_unexplored)

                for client_id in selected_unexplored:
                    if client_id in self.unexplored_clients:
                        self.unexplored_clients.remove(client_id)

                selected_clients += selected_unexplored

        context.state["prng_state"] = random.getstate()

        logging.info("[Server] Pisces selected clients: %s", selected_clients)
        return selected_clients

    def on_reports_received(
        self, updates: List[SimpleNamespace], context: ServerContext
    ) -> None:
        """Update client utilities and detect outliers if robustness enabled."""
        if not updates:
            return

        for update in updates:
            client_id = update.client_id
            staleness = getattr(update, "staleness", 0.0)
            self.client_staleness.setdefault(client_id, []).append(staleness)

            if hasattr(update.report, "statistical_utility"):
                base_utility = update.report.statistical_utility
                staleness_factor = self._calculate_staleness_factor(client_id)
                self.client_utilities[client_id] = base_utility * staleness_factor

                if self.robustness:
                    start_round = getattr(update.report, "start_round", None)
                    if start_round is not None:
                        self.model_versions_clients_dict.setdefault(
                            start_round, []
                        ).append((client_id, base_utility))
                        self._maybe_detect_outliers(start_round)

    def _maybe_detect_outliers(self, start_version: int) -> None:
        """Pool recent utilities and trigger anomaly detection if enough data."""
        tuples: List[Tuple[int, float]] = []
        already_existing_clients = set()

        for offset in range(self.augmented_factor):
            version = start_version - offset
            if version <= 0:
                break
            if version not in self.model_versions_clients_dict:
                continue

            current_records = []
            for client_id, loss_norm in self.model_versions_clients_dict[version]:
                if client_id in already_existing_clients:
                    continue
                already_existing_clients.add(client_id)
                current_records.append((client_id, loss_norm))
            tuples += current_records

        if len(tuples) >= self.threshold_factor * max(1, self.per_round):
            logging.info(
                "PiscesSelection: running anomaly detection with %d records.",
                len(tuples),
            )
            self._detect_outliers(tuples)
        else:
            logging.info(
                "PiscesSelection: insufficient records (%d) for anomaly detection.",
                len(tuples),
            )

    def _detect_outliers(self, tuples: List[tuple]) -> None:
        """Detect outliers via DBSCAN and update reliability credits."""
        if not tuples:
            return

        client_id_list = [item[0] for item in tuples]
        loss_list = np.array([item[1] for item in tuples]).reshape(-1, 1)

        min_samples = max(1, self.per_round // 2)
        eps = 0.5

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(loss_list)
        outliers = [
            client_id_list[idx]
            for idx, label in enumerate(clustering.labels_)
            if label == -1
        ]

        if not outliers:
            logging.info("PiscesSelection: no new outliers detected.")
            return

        newly_detected = []
        for client_id in outliers:
            current_credit = self.reliability_credit_record.get(
                client_id, self.reliability_credit_initial
            )
            current_credit = max(0, current_credit - 1)
            self.reliability_credit_record[client_id] = current_credit

            if current_credit == 0 and client_id not in self.detected_corrupted_clients:
                self.detected_corrupted_clients.append(client_id)
                newly_detected.append(client_id)

        if newly_detected:
            logging.info(
                "PiscesSelection: detected corrupted clients %s", newly_detected
            )

    def _calculate_staleness_factor(self, client_id: int) -> float:
        """Calculate staleness factor mirroring Pisces aggregation."""
        history = self.client_staleness.get(client_id, [])
        if not history:
            return 1.0

        recent_history = history[-self.history_window :]
        staleness = float(np.mean(recent_history))
        return 1.0 / pow(staleness + 1.0, self.staleness_factor)


class PolarisSelectionStrategy(ClientSelectionStrategy):
    """
    Polaris client selection using geometric programming to optimize sampling probabilities.

    Balances aggregation variance and staleness by solving a convex optimization problem
    (requires CVXOPT with the MOSEK backend).
    """

    def __init__(self, beta: float = 1.0, staleness_weight: float = 1.0):
        super().__init__()
        self.beta = beta
        self.staleness_weight = staleness_weight
        self.total_clients = 0
        self.local_gradient_bounds: np.ndarray | None = None
        self.local_stalenesses: np.ndarray | None = None
        self.aggregation_weights: np.ndarray | None = None

    def setup(self, context: ServerContext) -> None:
        """Initialize Polaris selection state."""
        self._ensure_solver_available()

        try:
            server_cfg = Config().server
            if hasattr(server_cfg, "polaris_beta"):
                self.beta = server_cfg.polaris_beta
            if hasattr(server_cfg, "polaris_staleness_weight"):
                self.staleness_weight = server_cfg.polaris_staleness_weight
        except ValueError:
            # Config not initialized, fall back to constructor defaults
            pass

        self.total_clients = context.total_clients
        polaris_state = context.state.setdefault("polaris", {})

        if "local_gradient_bounds" not in polaris_state:
            polaris_state["local_gradient_bounds"] = np.full(
                self.total_clients, 0.5, dtype=float
            )
        if "local_stalenesses" not in polaris_state:
            polaris_state["local_stalenesses"] = np.full(
                self.total_clients, 0.01, dtype=float
            )
        if "aggregation_weights" not in polaris_state:
            polaris_state["aggregation_weights"] = np.full(
                self.total_clients, 1.0 / max(1, self.total_clients), dtype=float
            )

        self.local_gradient_bounds = polaris_state["local_gradient_bounds"]
        self.local_stalenesses = polaris_state["local_stalenesses"]
        self.aggregation_weights = polaris_state["aggregation_weights"]

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        """Sample clients according to optimized probability distribution."""
        self._ensure_solver_available()

        assert clients_count <= len(clients_pool), (
            f"Cannot select {clients_count} clients from pool of {len(clients_pool)}"
        )

        # Refresh state references in case aggregator recreated them
        polaris_state = context.state.setdefault("polaris", {})
        self.local_gradient_bounds = polaris_state.get("local_gradient_bounds", None)
        self.local_stalenesses = polaris_state.get("local_stalenesses", None)
        self.aggregation_weights = polaris_state.get("aggregation_weights", None)

        if (
            self.local_gradient_bounds is None
            or self.local_stalenesses is None
            or self.aggregation_weights is None
        ):
            raise RuntimeError("PolarisSelection: required state arrays are missing.")

        probabilities = self._calculate_selection_probability(clients_pool)

        prng_state = context.state.get("prng_state")
        if prng_state:
            random.setstate(prng_state)

        selected_clients = np.random.choice(
            clients_pool, clients_count, replace=False, p=probabilities
        ).tolist()

        context.state["prng_state"] = random.getstate()

        logging.info("[Server] Polaris selected clients: %s", selected_clients)
        return selected_clients

    def on_reports_received(
        self, updates: List[SimpleNamespace], context: ServerContext
    ) -> None:
        """Update staleness and aggregation weights from received reports."""
        if not updates:
            return

        polaris_state = context.state.setdefault("polaris", {})
        self.local_stalenesses = polaris_state.setdefault(
            "local_stalenesses",
            np.full(self.total_clients, 0.01, dtype=float),
        )
        self.aggregation_weights = polaris_state.setdefault(
            "aggregation_weights",
            np.full(self.total_clients, 1.0 / max(1, self.total_clients), dtype=float),
        )

        total_samples = sum(update.report.num_samples for update in updates)
        for update in updates:
            client_index = update.client_id - 1
            staleness = getattr(update, "staleness", 0.0)
            self.local_stalenesses[client_index] = staleness + 0.1

            if total_samples > 0:
                self.aggregation_weights[client_index] = (
                    update.report.num_samples / total_samples
                )

    def _ensure_solver_available(self) -> None:
        """Ensure optional optimization dependencies are available."""
        if (
            mosek is None
            or solvers is None
            or matrix is None
            or sparse is None
            or log is None
        ):
            raise ImportError(
                "PolarisSelectionStrategy requires 'mosek' and 'cvxopt' to be installed."
            )

    def _calculate_selection_probability(self, clients_pool: List[int]) -> np.ndarray:
        """Solve the geometric program defining Polaris sampling probabilities."""
        clients_pool_zero_indexed = [client_id - 1 for client_id in clients_pool]
        num_of_clients = len(clients_pool_zero_indexed)

        aggregation_weights_inpool = self.aggregation_weights[clients_pool_zero_indexed]
        local_gradient_bounds_inpool = self.local_gradient_bounds[
            clients_pool_zero_indexed
        ]
        local_staleness_inpool = np.square(
            self.local_stalenesses[clients_pool_zero_indexed]
        )

        aggre_weight_square = np.square(aggregation_weights_inpool)
        local_gradient_bound_square = np.square(local_gradient_bounds_inpool)

        f1_params = matrix(
            self.beta * np.multiply(aggre_weight_square, local_gradient_bound_square)
        )

        f2_temp = np.multiply(local_staleness_inpool, local_gradient_bounds_inpool)
        f2_params = matrix(
            self.staleness_weight * np.multiply(aggre_weight_square, f2_temp)
        )

        f1 = matrix(-1.0 * np.eye(num_of_clients))
        f2 = matrix(np.eye(num_of_clients))
        F = sparse([[f1, f2]])

        g = log(matrix(sparse([[f1_params, f2_params]])))

        K = [2 * num_of_clients]
        G = matrix(-1.0 * np.eye(num_of_clients))
        h = matrix(np.zeros((num_of_clients, 1)))

        # Equality constraint sum(q_i) = 1
        A = matrix([[1.0]])
        if num_of_clients > 1:
            A1 = matrix([[1.0]])
            for _ in range(num_of_clients - 1):
                A = sparse([[A], [A1]])
        b = matrix([1.0])

        solvers.options["maxiters"] = 500
        solution = solvers.gp(
            K, F, g, G, h, A, b, solver="mosek" if mosek is not None else None
        )["x"]

        probabilities = np.array(solution, dtype=float).reshape(-1)
        probabilities = probabilities / probabilities.sum()
        return probabilities
