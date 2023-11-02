"""
Pisces: an asynchronous client selection and server aggregation algorithm.

Reference:

Z. Jiang, B. Wang, B. Li, B. Li. "Pisces: Efficient Federated Learning via Guided Asynchronous
Training," in Proceedings of ACM Symposium on Cloud Computing (SoCC), 2022.

URL: https://arxiv.org/abs/2206.09264
"""

import asyncio
import logging
import random

import numpy as np
from sklearn.cluster import DBSCAN

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the Pisces algorithm."""

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

        self.staleness_factor = Config().server.staleness_factor
        self.client_utilities = {}
        self.client_staleness = {}
        self.total_samples = 0

        # Exploration vs exploitation
        self.exploration_factor = Config().server.exploration_factor
        self.exploration_decaying_factor = Config().server.exploration_decaying_factor
        self.min_explore_factor = Config().server.min_explore_factor
        self.explored_clients = []
        self.unexplored_clients = []
        self.prng_state = random.getstate()

        # Below are for robustness
        self.robustness = False
        self.augmented_factor = 5
        self.threshold_factor = 1
        self.model_versions_clients_dict = {}
        self.per_round = Config().clients.per_round
        self.reliability_credit_record = {
            client_id: 5 for client_id in range(1, self.total_clients + 1)
        }
        self.detected_corrupted_clients = []

    def configure(self) -> None:
        """Initialize necessary variables."""
        super().configure()

        self.client_utilities = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_staleness = {
            client_id: [] for client_id in range(1, self.total_clients + 1)
        }
        self.unexplored_clients = list(range(1, self.total_clients + 1))

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging with calcuated staleness factor."""
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples
            self.client_staleness[updates[i].client_id].append(updates[i].staleness)

            staleness_factor = self._calculate_staleness_factor(updates[i].client_id)

            for name, delta in update.items():
                # Use weighted average by the number of samples and staleness factor
                avg_update[name] += (
                    delta * (num_samples / self.total_samples) * staleness_factor
                )

            # Yield to other tasks in the server
            await asyncio.sleep(0)
        return avg_update

    def _calculate_staleness_factor(self, client_id):
        """Calculate the client staleness factor."""
        stalenss = np.mean(self.client_staleness[client_id][-5:])
        return 1.0 / pow(stalenss + 1, self.staleness_factor)

    def weights_aggregated(self, updates):
        """Calculate client utility and update the record on the server"""
        for update in updates:
            self.client_utilities[
                update.client_id
            ] = update.report.statistical_utility * self._calculate_staleness_factor(
                update.client_id
            )

            if self.robustness:
                # Start to do pooling
                start_version = update.report.start_round
                if start_version not in self.model_versions_clients_dict:
                    self.model_versions_clients_dict[start_version] = [
                        (update.client_id, update.report.statistical_utility)
                    ]
                else:
                    self.model_versions_clients_dict[start_version].append(
                        (update.client_id, update.report.statistical_utility)
                    )

                tuples = []
                already_existing_clients = set()
                for i in range(self.augmented_factor):
                    if start_version - i <= 0:
                        break

                    # avoid cybil attacks, i.e., outliers repeat deliberately
                    tmp = []
                    for client_id, loss_norm in self.model_versions_clients_dict[
                        start_version - i
                    ]:
                        if client_id in already_existing_clients:
                            continue
                        already_existing_clients.add(client_id)
                        tmp.append((client_id, loss_norm))
                    tuples += tmp

                if len(tuples) >= self.threshold_factor * self.per_round:
                    logging.info(
                        "Starting anomaly detection with %s recent records.",
                        len(tuples),
                    )
                    self._detect_outliers(tuples)
                else:
                    logging.info(
                        "Records collected for anomaly detection are not enough: %s.",
                        len(tuples),
                    )

    def _detect_outliers(self, tuples):
        """Detect outliers from client updates by DBSCAN."""

        client_id_list = [tu[0] for tu in tuples]
        loss_list = [tu[1] for tu in tuples]
        loss_list = np.array(loss_list).reshape(-1, 1)
        min_samples = self.per_round // 2
        eps = 0.5

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(loss_list)
        result = clustering.labels_.tolist()
        outliers = [client_id_list[idx] for idx, e in enumerate(result) if e == -1]

        newly_detected_outliers = []
        for client_id in outliers:
            self.reliability_credit_record[client_id] -= 1
            if client_id not in self.detected_corrupted_clients:
                current_credit = self.reliability_credit_record[client_id]
                if current_credit == 0:
                    self.detected_corrupted_clients.append(client_id)
                    newly_detected_outliers.append(client_id)

        if len(newly_detected_outliers) == 0:
            logging.info("No new outliers.")
        else:
            newly_detected_outliers = sorted(newly_detected_outliers)

    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        selected_clients = []

        if self.robustness:
            outliers = [
                client_id
                for client_id in available_clients
                if client_id in self.detected_corrupted_clients
            ]
            available_clients = [
                client_id
                for client_id in available_clients
                if client_id not in self.detected_corrupted_clients
            ]
            logging.info(
                "These clients are detected as outliers and precluded from selection: %s",
                {outliers},
            )

        if self.current_round > 1:
            # Exploitation
            explored_clients_count = min(
                len(self.unexplored_clients),
                np.random.binomial(clients_count, self.exploration_factor, 1)[0],
            )

            self.exploration_factor = max(
                self.exploration_factor * self.exploration_decaying_factor,
                self.min_explore_factor,
            )

            exploited_clients_count = min(
                len(self.explored_clients), clients_count - explored_clients_count
            )

            sorted_by_utility = sorted(
                self.client_utilities, key=self.client_utilities.get, reverse=True
            )
            sorted_by_utility = [
                client for client in sorted_by_utility if client in clients_pool
            ]

            selected_clients = sorted_by_utility[:exploited_clients_count]

        # Exploration
        random.setstate(self.prng_state)

        # Select unexplored clients randomly
        selected_unexplored_clients = random.sample(
            self.unexplored_clients, clients_count - len(selected_clients)
        )

        self.prng_state = random.getstate()
        self.explored_clients += selected_unexplored_clients

        for client_id in selected_unexplored_clients:
            self.unexplored_clients.remove(client_id)

        selected_clients += selected_unexplored_clients

        logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients
