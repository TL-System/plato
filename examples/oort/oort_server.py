"""
A federated learning server using oort client selection.
"""

import logging
import math
import random
from collections import OrderedDict
import numpy as np

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """A federated learning server using oort client selection."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # Clients that will no longer be selected for future rounds.
        self.blacklist = []

        # All client utilities
        self.client_utilities = {}

        # Client training times
        self.client_durations = {}

        # The desired duration for each communication round
        self.desired_duration = Config().server.desired_duration

        self.explored_clients = []
        self.unexplored_clients = []

        # Keep track of each client's last participated round.
        self.client_last_rounds = {}

        self.exploration_factor = Config().server.exploration_factor
        self.step_window = Config().server.step_window
        self.pacer_step = Config().server.desired_duration

        self.penalty = Config().server.penalty

        # Keep track of statistical utility history.
        self.util_history = []

        # Number of times that each client has been selected
        self.times_selected = OrderedDict()

        # Cut off for sampling client utilities
        self.cut_off = (
            Config().server.cut_off if hasattr(Config().server, "cut_off") else 0.95
        )

        # Times should can a client be selected before being blacklisted?
        self.blacklist_num = (
            Config().server.blacklist_num
            if hasattr(Config().server, "blacklist_num")
            else 10
        )

    def configure(self):
        """Initialize necessary variables."""
        super().configure()
        self.client_durations = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_last_rounds = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.times_selected = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_utilities = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.unexplored_clients = list(range(1, self.total_clients + 1))

    def weights_aggregated(self, updates):
        """
        After the updated weights have been aggregated, extracts statistical
        utility, training times and adjusts desired round duration.
        """

        # Extract statistical utility and local training times
        for update in updates:
            self.client_utilities[update.client_id] = update.report.statistics_utility
            self.client_durations[update.client_id] = update.report.training_time
            self.client_last_rounds[update.client_id] = self.current_round

        # Calculate updated client utilities on explored clients
        for update in updates:
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
            if self.times_selected[update.client_id] > self.blacklist_num:
                self.blacklist.append(update.client_id)

    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        # Exploitation
        exploit_len = math.ceil((1.0 - self.exploration_factor) * clients_count)

        # If there aren't enough unexplored clients for exploration
        if (clients_count - exploit_len) > len(self.unexplored_clients):
            exploit_len = clients_count - len(self.unexplored_clients)

        # Take the top-k, sample by probability, take 95% of the cut-off loss by default
        sorted_util = sorted(
            self.client_utilities, key=self.client_utilities.get, reverse=True
        )

        # Take cut-off utility
        cut_off_util = (
            self.client_utilities[sorted_util[exploit_len - 1]] * self.cut_off
        )

        # Admit clients with utilities higher than the cut-off
        exploit_clients = []
        for client_id in sorted_util:
            if (
                self.client_utilities[client_id] > cut_off_util
                and client_id not in self.blacklist
            ):
                exploit_clients.append(client_id)

        last_index = (
            0 if len(exploit_clients) == 0 else sorted_util.index(exploit_clients[-1])
        )

        # Sample by utiliity probability.
        total_sc = max(
            1e-4,
            float(sum([self.client_utilities[key] for key in exploit_clients])),
        )
        probabilities = [
            self.client_utilities[key] / total_sc for key in exploit_clients
        ]

        selected_clients = []
        if len(exploit_clients) < exploit_len:
            num = len(exploit_clients)
        else:
            num = exploit_len

        if len(probabilities) != 0 and exploit_len != 0:
            selected_clients = np.random.choice(
                exploit_clients, num, p=probabilities, replace=False
            )
            selected_clients = selected_clients.tolist()

        # If the result of exploitation wasn't enough to meet the required length
        if len(selected_clients) < exploit_len and self.current_round > 1:
            for step in range(last_index + 1, len(sorted_util)):
                if (
                    not sorted_util[step] in self.blacklist
                    and len(selected_clients) != exploit_len
                ):
                    selected_clients.append(sorted_util[step])

        # Exploration
        explore_clients = []
        random.setstate(self.prng_state)

        # Select unexplored clients randomly
        explore_clients = random.sample(
            self.unexplored_clients, clients_count - len(selected_clients)
        )

        self.prng_state = random.getstate()
        self.explored_clients += explore_clients

        self.unexplored_clients = [
            id for id in self.unexplored_clients if id not in explore_clients
        ]

        selected_clients += explore_clients

        for client in selected_clients:
            self.times_selected[client] += 1

        logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients

    def calc_client_util(self, client_id):
        """Calculate client utility."""
        # Set temporal uncertainty
        if self.client_last_rounds[client_id] != 0:
            temp_uncertainty = math.sqrt(
                0.1 * math.log(self.current_round) / self.client_last_rounds[client_id]
            )
        else:
            temp_uncertainty = 0
        client_utility = self.client_utilities[client_id] + temp_uncertainty

        if self.desired_duration < self.client_durations[client_id]:
            global_utility = (
                self.desired_duration / self.client_durations[client_id]
            ) ** self.penalty
            client_utility *= global_utility

        return client_utility
