"""
A federated learning server using oort client selection.
"""

import logging
import math
import os
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
        self.client_utilities = OrderedDict()

        # Client training times
        self.client_durations = [None]

        # The desired duration for each communication round
        self.desired_duration = Config().server.desired_duration

        self.explored_clients = []
        self.unexplored_clients = []

        # Keep track of each client's last involved round.
        self.last_round = [0]

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
        self.client_durations = self.client_durations * self.total_clients
        self.last_round = self.last_round * self.total_clients
        self.times_selected = {num: 0 for num in range(1, self.total_clients + 1)}
        self.unexplored_clients = [
            client_id for client_id in range(1, self.total_clients + 1)
        ]
        self.client_utilities = {num: 0 for num in range(1, self.total_clients + 1)}

    def weights_aggregated(self, updates):
        """
        After the updated weights have been aggregated, extracts statistical
        utility, training times and adjusts desired round duration.
        """

        # Extract statistical utility and local training times
        for (client_id, report, __, __) in updates:
            self.client_utilities[client_id] = report.statistics_utility
            self.client_durations[client_id - 1] = report.training_time
            self.last_round[client_id - 1] = self.current_round

        # Calculate updated client utilities on explored clients
        for (client_id, __, __, __) in updates:
            self.client_utilities[client_id] = self.calc_client_util(client_id)

        # Adjust pacer
        self.util_history.append(self.calc_util_sum(updates))
        if self.current_round >= 2 * self.step_window:
            last_pacer_rounds = sum(
                self.util_history[-2 * self.step_window : -self.step_window]
            )
            current_pacer_rounds = sum(self.util_history[-self.step_window :])
            if last_pacer_rounds > current_pacer_rounds:
                self.desired_duration += self.pacer_step

        # Blacklist clients who have been selected self.blacklist_num times
        for (client_id, __, __, __) in updates:
            if self.times_selected[client_id] > self.blacklist_num:
                self.blacklist.append(client_id)

    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        # Exploitation
        exploit_len = math.ceil((1.0 - self.exploration_factor) * clients_count)

        # If there aren't enough unexplored clients for exploration.
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
        if self.last_round[client_id - 1] != 0:
            temp_uncertainty = math.sqrt(
                0.1 * math.log(self.current_round) / self.last_round[client_id - 1]
            )
        else:
            temp_uncertainty = 0
        client_utility = self.client_utilities[client_id] + temp_uncertainty

        if self.desired_duration < self.client_durations[client_id - 1]:
            global_utility = (
                self.desired_duration / self.client_durations[client_id - 1]
            ) ** self.penalty
            client_utility *= global_utility

        return client_utility

    def calc_util_sum(self, updates):
        """Calculate sum of statistical utilities from client reports."""
        total = 0
        for (__, report, __, __) in updates:
            total += report.statistics_utility

        return total

    def server_will_close(self):
        """
        Method called at the start of closing the server.
        """
        model_name = Config().trainer.model_name
        model_path = Config().params["checkpoint_path"]
        for client_id in range(1, self.total_clients + 1):
            loss_path = f"{model_path}/{model_name}_{client_id}_squared_batch_loss.pth"
            if os.path.exists(loss_path):
                os.remove(loss_path)
