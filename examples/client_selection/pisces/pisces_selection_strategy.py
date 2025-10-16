"""
Pisces client selection strategy.
"""

from __future__ import annotations

import logging
import random
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from plato.config import Config
from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext


class PiscesSelectionStrategy(ClientSelectionStrategy):
    """Select clients with staleness-aware utilities and optional robustness."""

    def __init__(
        self,
        exploration_factor: float = 0.3,
        exploration_decaying_factor: float = 0.99,
        min_explore_factor: float = 0.1,
        staleness_factor: float = 1.0,
        robustness: bool = False,
        augmented_factor: int = 5,
        threshold_factor: float = 1.0,
        speed_penalty_factor: float = 0.5,
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
        self.speed_penalty_factor = speed_penalty_factor
        self.reliability_credit_initial = reliability_credit_initial
        self.history_window = history_window

        self.client_utilities: dict[int, float] = {}
        self.client_staleness: dict[int, List[float]] = {}
        self.explored_clients: List[int] = []
        self.unexplored_clients: List[int] = []
        self.reliability_credit_record: dict[int, int] = {}
        self.detected_corrupted_clients: List[int] = []
        self.model_versions_clients_dict: dict[int, List[Tuple[int, float]]] = {}
        self.client_last_latency: dict[int, float] = {}
        self.per_round = 0

    def setup(self, context: ServerContext) -> None:
        """Initialise state dictionaries and allow config overrides."""
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
            if hasattr(server_cfg, "speed_penalty_factor"):
                self.speed_penalty_factor = server_cfg.speed_penalty_factor
            if hasattr(server_cfg, "history_window"):
                self.history_window = server_cfg.history_window
        except ValueError:
            pass

        total_clients = context.total_clients
        self.client_utilities = {
            client_id: 0.0 for client_id in range(1, total_clients + 1)
        }
        self.client_staleness = {
            client_id: [] for client_id in range(1, total_clients + 1)
        }
        self.unexplored_clients = list(range(1, total_clients + 1))
        self.explored_clients = []
        self.reliability_credit_record = {
            client_id: self.reliability_credit_initial
            for client_id in range(1, total_clients + 1)
        }
        self.detected_corrupted_clients = []
        self.model_versions_clients_dict = {}
        self.client_last_latency = {
            client_id: 1.0 for client_id in range(1, total_clients + 1)
        }
        self.per_round = context.clients_per_round

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        """Select clients using Pisces exploration/exploitation policy."""
        assert clients_count <= len(clients_pool)

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

        score_dict = self._compute_scores(available_clients, effective_count)

        if current_round > 1:
            unexplored_available = [
                client_id
                for client_id in self.unexplored_clients
                if client_id in available_clients
            ]
            exploration_quota = min(
                len(unexplored_available),
                np.random.binomial(effective_count, self.exploration_factor, 1)[0],
            )

            self.exploration_factor = max(
                self.exploration_factor * self.exploration_decaying_factor,
                self.min_explore_factor,
            )

            explored_available = [
                client_id
                for client_id in self.explored_clients
                if client_id in available_clients
            ]

            exploited_clients_target = max(0, effective_count - exploration_quota)
            exploited_clients_count = min(
                len(explored_available), exploited_clients_target
            )

            explored_sorted = sorted(
                explored_available,
                key=lambda client: score_dict.get(client, 0.0),
                reverse=True,
            )
            selected_clients = explored_sorted[:exploited_clients_count]
        else:
            exploration_quota = min(effective_count, len(self.unexplored_clients))

        prng_state = context.state.get("prng_state")
        if prng_state:
            random.setstate(prng_state)

        remaining_slots = effective_count - len(selected_clients)
        if remaining_slots > 0 and exploration_quota > 0:
            exploration_candidates = [
                client_id
                for client_id in self.unexplored_clients
                if client_id in available_clients and client_id not in selected_clients
            ]
            explore_count = min(
                exploration_quota, remaining_slots, len(exploration_candidates)
            )

            if explore_count > 0:
                selected_unexplored = random.sample(
                    exploration_candidates, explore_count
                )
                for client_id in selected_unexplored:
                    self.explored_clients.append(client_id)
                    if client_id in self.unexplored_clients:
                        self.unexplored_clients.remove(client_id)

                selected_clients.extend(selected_unexplored)

        if len(selected_clients) < effective_count:
            remaining_candidates = [
                client_id
                for client_id in available_clients
                if client_id not in selected_clients
            ]
            fill_count = min(
                effective_count - len(selected_clients), len(remaining_candidates)
            )
            if fill_count > 0:
                selected_clients.extend(random.sample(remaining_candidates, fill_count))

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

            latency = (
                getattr(update.report, "training_time", 0.0)
                + getattr(update.report, "processing_time", 0.0)
                + getattr(update.report, "comm_time", 0.0)
            )
            if latency and latency > 0:
                self.client_last_latency[client_id] = float(latency)

            if hasattr(update.report, "statistical_utility"):
                base_utility = update.report.statistical_utility
                self.client_utilities[client_id] = base_utility

                if client_id not in self.explored_clients:
                    self.explored_clients.append(client_id)
                if client_id in self.unexplored_clients:
                    self.unexplored_clients.remove(client_id)

                if self.robustness:
                    start_round = getattr(update.report, "start_round", None)
                    if start_round is not None:
                        self.model_versions_clients_dict.setdefault(
                            start_round, []
                        ).append((client_id, base_utility))
                        self._maybe_detect_outliers(start_round)

    def _compute_scores(
        self, available_clients: List[int], selection_count: int
    ) -> dict[int, float]:
        """Compute combined utility scores including speed and staleness penalties."""
        if self.per_round > 0:
            dynamic_speed_factor = self.speed_penalty_factor * (
                1 - selection_count / self.per_round
            )
        else:
            dynamic_speed_factor = self.speed_penalty_factor

        scores: dict[int, float] = {}
        for client_id in available_clients:
            score = self.client_utilities.get(client_id, 0.0)

            if dynamic_speed_factor > 0:
                latency = self.client_last_latency.get(client_id, 1.0)
                if latency <= 0:
                    latency = 1.0
                score *= (1.0 / latency) ** dynamic_speed_factor

            staleness_penalty = self._calculate_staleness_factor(client_id)
            score *= staleness_penalty

            scores[client_id] = score
        return scores

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
            current_credit = current_credit - 1
            self.reliability_credit_record[client_id] = current_credit

            if current_credit <= 0 and client_id not in self.detected_corrupted_clients:
                self.detected_corrupted_clients.append(client_id)
                newly_detected.append(client_id)

        if newly_detected:
            newly_detected.sort()
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
