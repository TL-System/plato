"""
An asynchronous federated learning server using Sirius.
"""

import math
import time
import random
import logging
import asyncio
import numpy as np
from plato.config import Config
from plato.servers import fedavg
from sklearn.cluster import DBSCAN

from inspect import currentframe

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

class Server(fedavg.Server):
    """A federated learning server using the sirius algorithm."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.staleness_factor = Config().server.staleness_factor
        self.exploration_factor = Config().server.exploration_factor
        self.exploration_decaying_factor = Config().server.exploration_decaying_factor
        self.min_explore_factor = Config().server.min_explore_factor
        self.explored_clients = []
        self.unexplored_clients = []
        print("Server finished line_", get_linenumber())

    def configure(self):
        """Initialize necessary variables."""
        super().configure()

        self.client_utilities = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.unexplored_clients = list(range(1, self.total_clients + 1))
        print("Server finished line_", get_linenumber())

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging with calcuated staleness factor."""
        print("Server finished line_", get_linenumber()) 
        
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
            staleness = updates[i].staleness
            staleness_factor = self.staleness_function(staleness)

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples) * staleness_factor # no normalization but from Zhifeng's code.

            # Yield to other tasks in the server
            await asyncio.sleep(0)
        print("Server finished line_", get_linenumber())
        return avg_update

    def staleness_function(self, stalenss):
        print("Server finished line_", get_linenumber())
        return 1.0 / pow(stalenss + 1, self.staleness_factor) # formula obtained from Zhifeng's code. (clients_manager/sirius/staleness_factor_calculator)

    def weights_aggregated(self, updates):
        """Method called at the end of aggregating received weights."""
        """Calculate client utility here and update the record on the server"""
        for update in updates:
            self.client_utilities[update.client_id] = update.report.statistics_utility * self.staleness_function(update.staleness)
        # Start to do pooling
        if len(tuples) >= threshold_factor * self.client_per_round: #凑够了就开始detect。
            logging.info(f"Starting anomaly detection with {len(tuples)} recent records.")
            self.detect_outliers(tuples)
        else:
            logging.info(f"Records collected for anomaly detection are not enough: {len(tuples)}.")

    def detect_outliers(self, tuples):
        start_time = time.perf_counter()

        client_id_list = [tu[0] for tu in tuples]
        loss_list = [tu[1] for tu in tuples]
        loss_list = np.array(loss_list).reshape(-1, 1)
        min_samples = self.client_per_round // 2  # TODO: avoid hard-coding
        eps = 0.5
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(loss_list)
        result = clustering.labels_.tolist()
        outliers = [client_id_list[idx]
                    for idx, e in enumerate(result) if e == -1]
        debug_dict = {
            'client_id_list': client_id_list,
            'loss_list': loss_list.squeeze(-1),  # for ease of reading
            'DBSCAN_res': result
        }
        logging.info(f"[Debug] debug_dict for DBSCAN: {debug_dict}.")
        logging.info(f"[Debug] Note actual outliers: {self.expected_corrupted_clients}.")

        end_time = time.perf_counter()
        duration = round(end_time - start_time, 2)
        logging.info(f"Outliers detected by DBSCAN "
                     f"in {duration} sec: {outliers}.")

        newly_detected_outliers = []
        for client_id in outliers:
            self.reliability_credit_record[client_id] -= 1
            if client_id not in self.detected_corrupted_clients:
                current_credit = self.reliability_credit_record[client_id]
                if current_credit == 0:
                    self.detected_corrupted_clients.append(client_id)
                    newly_detected_outliers.append(client_id)

        if len(newly_detected_outliers) == 0:
            logging.info(f"No new outliers.")
        else:
            newly_detected_outliers = sorted(newly_detected_outliers )
            logging.info(f"{len(newly_detected_outliers)} clients "
                         f"are newly taken as outliers"
                         f": {newly_detected_outliers}.")

    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        selected_clients = []
        print("Server finished line_", get_linenumber())
        if self.current_round > 1:
            # Exploitation
            num_to_explore = min(
                len(self.unexplored_clients),
                np.random.binomial(clients_count, self.exploration_factor, 1)[0]) # ??

            self.exploration_factor = max(
                self.exploration_factor * self.exploration_decaying_factor,
                self.min_explore_factor)
            
            real_exploit_num = min(len(self.explored_clients),
                                     clients_count - num_to_explore)

            print("self.exploration_factor: ", self.exploration_factor)    

            
            print("exploit_client count: ", real_exploit_num)
            sorted_util = sorted(
                self.client_utilities, key=self.client_utilities.get, reverse=True
            )
           
            print("Client utilities are: ", sorted_util)
            selected_clients = sorted_util[:real_exploit_num]
            print("Exploited clients are: ", selected_clients)
        print("Server finished line_", get_linenumber())
        # Exploration
        random.setstate(self.prng_state)

        # Select unexplored clients randomly
        selected_unexplore_clients = random.sample(
            self.unexplored_clients, clients_count - len(selected_clients))

        print("Randomly selected clients are: ", selected_unexplore_clients)
        self.prng_state = random.getstate()
        self.explored_clients += selected_unexplore_clients

        for client_id in selected_unexplore_clients:
            self.unexplored_clients.remove(client_id)
        print("Server finished line_", get_linenumber())
        selected_clients += selected_unexplore_clients

        #for client in selected_clients:
        #    self.client_selected_times[client] += 1

        logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients
           
