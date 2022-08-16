"""
A federated learning server using oort client selection.
"""

import logging
import math
import random
import numpy as np

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """A federated learning server using oort client selection."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # Clients that will no longer be selected for future rounds.
        self.blacklist = []

        # All clients' utilities
        self.client_utilities = {}

        # All clients‘ training times
        self.client_durations = {}

        # Keep track of each client's last participated round.
        self.client_last_rounds = {}

        # Number of times that each client has been selected
        self.client_selected_times = {}

        # The desired duration for each communication round
        self.desired_duration = Config().server.desired_duration

        self.explored_clients = []
        self.unexplored_clients = []

        self.exploration_factor = Config().server.exploration_factor
        self.step_window = Config().server.step_window
        self.pacer_step = Config().server.desired_duration

        self.penalty = Config().server.penalty

        # Keep track of statistical utility history.
        self.util_history = []

        # Cut off for sampling client utilities
        self.cut_off = (
            Config().server.cut_off if hasattr(Config().server, "cut_off") else 0.95
        )

        # Times a client is selected before being blacklisted
        self.blacklist_num = (
            Config().server.blacklist_num
            if hasattr(Config().server, "blacklist_num")
            else 10
        )

    def configure(self):
        """Initialize necessary variables."""
        super().configure()

        self.client_utilities = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_durations = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_last_rounds = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_selected_times = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }

        self.unexplored_clients = list(range(1, self.total_clients + 1))

    def weights_aggregated(self, updates):
        """Method called at the end of aggregating received weights."""
        for update in updates:
            # Extract statistical utility and local training times
            self.client_utilities[update.client_id] = update.report.statistics_utility
            self.client_durations[update.client_id] = update.report.training_time
            self.client_last_rounds[update.client_id] = self.current_round

            # Calculate client utilities of explored clients
            self.client_utilities[update.client_id] = self.calc_client_util(
                update.client_id
            )

        # Adjust pacer
        self.util_history.append(
            sum(update.report.statistics_utility for update in updates)
        )

        if self.current_round >= 2 * self.step_window:
            last_pacer_rounds = sum(
                self.util_history[-2 * self.step_window : -self.step_window]
            )
            current_pacer_rounds = sum(self.util_history[-self.step_window :])
            if last_pacer_rounds > current_pacer_rounds:
                self.desired_duration += self.pacer_step

        # Blacklist clients who have been selected self.blacklist_num times
        for update in updates:
            if self.client_selected_times[update.client_id] > self.blacklist_num:
                self.blacklist.append(update.client_id)

    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        selected_clients = []

        if self.current_round > 1:
            # Exploitation
            exploit_client_num = max(
                math.ceil((1.0 - self.exploration_factor) * clients_count),
                clients_count - len(self.unexplored_clients),
            )

            sorted_util = sorted(
                self.client_utilities, key=self.client_utilities.get, reverse=True
            )

            # Calculate cut-off utility
            cut_off_util = (
                self.client_utilities[sorted_util[exploit_client_num - 1]]
                * self.cut_off
            )

            # Include clients with utilities higher than the cut-off
            exploit_clients = []
            for client_id in sorted_util:
                if (
                    self.client_utilities[client_id] > cut_off_util
                    and client_id not in self.blacklist
                ):
                    exploit_clients.append(client_id)

            # Sample clients with their utilities
            utility_sum = float(
                sum([self.client_utilities[client_id] for client_id in exploit_clients])
            )

            probabilities = [
                self.client_utilities[client_id] / utility_sum
                for client_id in exploit_clients
            ]

            if len(probabilities) > 0 and exploit_client_num > 0:
                selected_clients = np.random.choice(
                    exploit_clients,
                    min(len(exploit_clients), exploit_client_num),
                    p=probabilities,
                    replace=False,
                )
                selected_clients = selected_clients.tolist()

            last_index = (
                sorted_util.index(exploit_clients[-1]) if exploit_clients else 0
            )

            # If the result of exploitation wasn't enough to meet the required length
            if len(selected_clients) < exploit_client_num:
                for index in range(last_index + 1, len(sorted_util)):
                    if (
                        not sorted_util[index] in self.blacklist
                        and len(selected_clients) < exploit_client_num
                    ):
                        selected_clients.append(sorted_util[index])

        # Exploration
        random.setstate(self.prng_state)

        # Select unexplored clients randomly
        selected_unexplore_clients = random.sample(
            self.unexplored_clients, clients_count - len(selected_clients)
        )

        self.prng_state = random.getstate()
        self.explored_clients += selected_unexplore_clients

        for client_id in selected_unexplore_clients:
            self.unexplored_clients.remove(client_id)

        selected_clients += selected_unexplore_clients

        for client in selected_clients:
            self.client_selected_times[client] += 1

        logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients

    def calc_client_util(self, client_id):
        """Calculate client utility."""
        client_utility = self.client_utilities[client_id] + math.sqrt(
            0.1 * math.log(self.current_round) / self.client_last_rounds[client_id]
        )

        if self.desired_duration < self.client_durations[client_id]:
            global_utility = (
                self.desired_duration / self.client_durations[client_id]
            ) ** self.penalty
            client_utility *= global_utility

        return client_utility
