"""
A federated learning server using oort client selection.
"""

import logging
import math
import pickle
import os
import sys
import random
from collections import OrderedDict
import numpy as np

from plato.servers import fedavg
from plato.config import Config
from plato.utils import fonts


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

        # Keeps track of each client's last involved round.
        self.last_round = [0]

        self.exploration_factor = Config().server.exploration_factor
        self.step_window = Config().server.step_window
        self.pacer_step = Config().server.desired_duration

        self.penalty = Config().server.penalty

        # Keeps track of statistical utility history.
        self.util_history = []

        # Number of times that each client has been selected
        self.times_selected = OrderedDict()

        # Cut off for sampling client utilities
        self.cut_off = Config().server.cut_off if hasattr(
            Config().server, "cut_off") else 0.95

        # How many times should can a client be selected before
        # being blacklisted?
        self.blacklist_num = Config().server.blacklist_num if hasattr(
            Config().server, "blacklist_num") else 10

    def configure(self):
        """ Initializes necessary variables. """
        super().configure()
        self.client_durations = self.client_durations * self.total_clients
        self.last_round = self.last_round * self.total_clients
        self.times_selected = {
            num: 0
            for num in range(1, self.total_clients + 1)
        }
        self.unexplored_clients = [
            client_id for client_id in range(1, self.total_clients + 1)
        ]
        self.client_utilities = {
            num: 0
            for num in range(1, self.total_clients + 1)
        }

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        deltas = await self.federated_averaging(updates)
        self.update_clients(updates)
        updated_weights = self.algorithm.update_weights(deltas)
        self.algorithm.load_weights(updated_weights)

    def update_clients(self, updates):
        """Extracts statistical utility, training times and adjusts desired round duration."""

        # Extracting statistical utility and local training times.
        for (client_id, report, __, __) in updates:
            self.client_utilities[client_id] = report.statistics_utility
            self.client_durations[client_id - 1] = report.training_time
            self.last_round[client_id - 1] = self.current_round

        # Calculating updated client utilities on explored clients
        for client_id in self.explored_clients:
            self.client_utilities[client_id] = self.calc_client_util(client_id)

        # Adjusts pacer
        self.util_history.append(self.calc_util_sum(self.updates))
        if self.current_round >= 2 * self.step_window:
            last_pacer_rounds = sum(
                self.util_history[-2 * self.step_window:-self.step_window])
            current_pacer_rounds = sum(self.util_history[-self.step_window:])
            if last_pacer_rounds > current_pacer_rounds:
                self.desired_duration += self.pacer_step

        # Blacklists clients who have been selected 10 times.
        for (client_id, __, __, __) in updates:
            if self.times_selected[client_id] > self.blacklist_num:
                self.blacklist.append(client_id)

    async def select_clients(self, for_next_batch=False):
        """ Select a subset of the clients and send messages to them to start training. """

        if not for_next_batch:
            # Selects clients according the oort algorithm

            # Exploitation
            exploit_len = math.ceil(
                (1.0 - self.exploration_factor) * self.clients_per_round)

            # If there aren't enough unexplored clients for exploration.
            if (self.clients_per_round - exploit_len) > len(
                    self.unexplored_clients):
                exploit_len = (self.clients_per_round -
                               len(self.unexplored_clients))

            # take the top-k, sample by probability, take 95% of the cut-off loss by default
            sorted_util = sorted(self.client_utilities,
                                 key=self.client_utilities.get,
                                 reverse=True)

            # take cut-off utility
            cut_off_util = self.client_utilities[sorted_util[exploit_len -
                                                             1]] * self.cut_off

            # Admit clients with utilities higher than the cut-off
            exploit_clients = []
            for client_id in sorted_util:
                if self.client_utilities[
                        client_id] > cut_off_util and client_id not in self.blacklist:
                    exploit_clients.append(client_id)

            if len(exploit_clients) != 0:
                last_index = sorted_util.index(exploit_clients[-1])
            else:
                last_index = 0

            # Sample by utiliity probability.
            total_sc = max(
                1e-4,
                float(
                    sum([
                        self.client_utilities[key] for key in exploit_clients
                    ])))
            probabilities = [
                self.client_utilities[key] / total_sc
                for key in exploit_clients
            ]

            picked_clients = []
            if len(exploit_clients) < exploit_len:
                num = len(exploit_clients)
            else:
                num = exploit_len

            if len(probabilities) != 0 and exploit_len != 0:
                picked_clients = np.random.choice(exploit_clients,
                                                  num,
                                                  p=probabilities,
                                                  replace=False)
                picked_clients = picked_clients.tolist()

            # If the result of exploitation wasn't enough to meet the required length.
            if len(picked_clients) < exploit_len and self.current_round != 0:
                for step in range(last_index + 1, len(sorted_util)):
                    if not sorted_util[step] in self.blacklist and len(
                            picked_clients) != exploit_len:
                        picked_clients.append(sorted_util[step])

            # Exploration
            explore_clients = []
            random.setstate(self.prng_state)

            # Select unexplored clients randomly
            explore_clients = random.sample(
                self.unexplored_clients,
                self.clients_per_round - len(picked_clients))

            self.prng_state = random.getstate()
            self.explored_clients += explore_clients

            self.unexplored_clients = [
                id for id in self.unexplored_clients
                if id not in explore_clients
            ]

            picked_clients += explore_clients

            for client in picked_clients:
                self.times_selected[client] += 1

            self.updates = []
            self.current_round += 1
            self.round_start_wall_time = self.wall_time

            if hasattr(Config().trainer, 'max_concurrency'):
                self.trained_clients = []

            logging.info(
                fonts.colourize(
                    f"\n[{self}] Starting round {self.current_round}/{Config().trainer.rounds}."
                ))

            logging.info("[%s] Selected clients: %s", self, picked_clients)

            if Config().is_central_server():
                # In cross-silo FL, the central server selects from the pool of edge servers
                self.clients_pool = list(self.clients)

            elif not Config().is_edge_server():
                self.clients_pool = list(range(1, 1 + self.total_clients))

            # In asychronous FL, avoid selecting new clients to replace those that are still
            # training at this time

            # When simulating the wall clock time, if len(self.reported_clients) is 0, the
            # server has aggregated all reporting clients already
            if self.asynchronous_mode and self.selected_clients is not None and len(
                    self.reported_clients) > 0 and len(
                        self.reported_clients) < self.clients_per_round:
                # If self.selected_clients is None, it implies that it is the first iteration;
                # If len(self.reported_clients) == self.clients_per_round, it implies that
                # all selected clients have already reported.

                # Except for these two cases, we need to exclude the clients who are still
                # training.
                training_client_ids = [
                    self.training_clients[client_id]['id']
                    for client_id in list(self.training_clients.keys())
                ]

                # If the server is simulating the wall clock time, some of the clients who
                # reported may not have been aggregated; they should be excluded from the next
                # round of client selection
                reporting_client_ids = [
                    client[2]['client_id'] for client in self.reported_clients
                ]

                selectable_clients = [
                    client for client in self.clients_pool
                    if client not in training_client_ids
                    and client not in reporting_client_ids
                ]

                # Selects clients according to the oort algorithm.
                if self.simulate_wall_time:
                    self.selected_clients = self.choose_clients(
                        selectable_clients,
                        len(self.current_processed_clients))
                else:
                    self.selected_clients = self.choose_clients(
                        selectable_clients, len(self.reported_clients))
            else:
                self.selected_clients = picked_clients

            self.current_reported_clients = {}
            self.current_processed_clients = {}

            # There is no need to clear the list of reporting clients if we are
            # simulating the wall clock time on the server. This is because
            # when wall clock time is simulated, the server needs to wait for
            # all the clients to report before selecting a subset of clients for
            # replacement, and all remaining reporting clients will be processed
            # in the next round
            if not self.simulate_wall_time:
                self.reported_clients = []

        if len(self.selected_clients) > 0:
            self.selected_sids = []

            # If max_concurrency is specified, run selected clients batch by batch,
            # and the number of clients in each batch (on each GPU, if multiple GPUs are available)
            # is equal to # (or maybe smaller than for the last batch) max_concurrency
            if hasattr(Config().trainer,
                       'max_concurrency') and not Config().is_central_server():
                selected_clients = []
                if Config().gpu_count() > 1:
                    untrained_clients = list(
                        set(self.selected_clients).difference(
                            self.trained_clients))
                    available_gpus = Config().gpu_count()
                    for cuda_id in range(available_gpus):
                        for client_id in untrained_clients:
                            if client_id % available_gpus == cuda_id:
                                selected_clients.append(client_id)
                            if len(selected_clients) >= min(
                                (cuda_id + 1) *
                                    Config().trainer.max_concurrency,
                                    self.clients_per_round):
                                break
                else:
                    selected_clients = self.selected_clients[
                        len(self.trained_clients):min(
                            len(self.trained_clients) + Config().trainer.
                            max_concurrency, len(self.selected_clients))]

                self.trained_clients += selected_clients

            else:
                selected_clients = self.selected_clients

            for i, selected_client_id in enumerate(selected_clients):
                self.selected_client_id = selected_client_id

                if Config().is_central_server():
                    client_id = selected_client_id
                elif Config().is_edge_server():
                    client_id = self.launched_clients[i]
                else:
                    client_id = i + 1

                sid = self.clients[client_id]['sid']

                if self.asynchronous_mode and self.simulate_wall_time:

                    # skip if this sid is currently `training' with reporting clients
                    # or it has already been selected in this round
                    while sid in self.training_sids or sid in self.selected_sids:
                        client_id = client_id % self.clients_per_round + 1
                        sid = self.clients[client_id]['sid']

                    self.training_sids.append(sid)
                    self.selected_sids.append(sid)

                self.training_clients[self.selected_client_id] = {
                    'id': self.selected_client_id,
                    'starting_round': self.current_round,
                    'start_time': self.round_start_wall_time,
                    'update_requested': False
                }

                logging.info("[%s] Selecting client #%d for training.", self,
                             self.selected_client_id)

                server_response = {
                    'id': self.selected_client_id,
                    'current_round': self.current_round
                }
                payload = self.algorithm.extract_weights()
                payload = self.customize_server_payload(payload)

                if self.comm_simulation:
                    logging.info(
                        "[%s] Sending the current model to client #%d (simulated).",
                        self, self.selected_client_id)

                    # First apply outbound processors, if any
                    payload = self.outbound_processor.process(payload)

                    model_name = Config().trainer.model_name if hasattr(
                        Config().trainer, 'model_name') else 'custom'
                    checkpoint_path = Config().params['checkpoint_path']

                    payload_filename = f"{checkpoint_path}/{model_name}_{self.selected_client_id}.pth"

                    with open(payload_filename, 'wb') as payload_file:
                        pickle.dump(payload, payload_file)

                    server_response['payload_filename'] = payload_filename

                    payload_size = sys.getsizeof(
                        pickle.dumps(payload)) / 1024**2

                    logging.info(
                        "[%s] Sending %.2f MB of payload data to client #%d (simulated).",
                        self, payload_size, self.selected_client_id)

                    self.comm_overhead += payload_size

                    # Compute the communication time to transfer the current global model to client
                    self.downlink_comm_time[
                        self.selected_client_id] = payload_size / (
                            self.downlink_bandwidth /
                            len(self.selected_clients))

                server_response = await self.customize_server_response(
                    server_response)

                # Sending the server response as metadata to the clients (payload to follow)
                await self.sio.emit('payload_to_arrive',
                                    {'response': server_response},
                                    room=sid)

                if not self.comm_simulation:
                    # Sending the server payload to the client
                    logging.info(
                        "[%s] Sending the current model to client #%d.", self,
                        selected_client_id)

                    await self.send(sid, payload, selected_client_id)

    def calc_client_util(self, client_id):
        """ Calculates client utility. """

        # temporal uncertainty.
        if self.last_round[client_id - 1] != 0:
            temp_uncertainty = math.sqrt(0.1 * math.log(self.current_round) /
                                         self.last_round[client_id - 1])
        else:
            temp_uncertainty = 0
        client_utility = self.client_utilities[client_id] + temp_uncertainty

        if self.desired_duration < self.client_durations[client_id - 1]:
            global_utility = (
                self.desired_duration /
                self.client_durations[client_id - 1])**self.penalty
            client_utility *= global_utility

        return client_utility

    def calc_util_sum(self, updates):
        """ Calculates sum of statistical utilities from client reports. """
        total = 0
        for (__, report, __, __) in updates:
            total += report.statistics_utility

        return total

    # pylint: disable=protected-access
    async def close(self):
        """ Closing the server. """
        logging.info("[%s] Training concluded.", self)
        self.trainer.save_model()

        # Delete gradient files created by the clients.
        model_name = Config().trainer.model_name
        model_path = Config().params['checkpoint_path']
        for client_id in range(1, self.total_clients + 1):
            loss_path = f"{model_path}/{model_name}_{client_id}__squred_batch_loss.pth"
            if os.path.exists(loss_path):
                os.remove(loss_path)

        await self.close_connections()
        os._exit(0)
