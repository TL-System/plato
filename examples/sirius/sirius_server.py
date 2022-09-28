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


class Server(fedavg.Server):
    """A federated learning server using the Sirius algorithm."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.staleness_factor = Config().server.staleness_factor

        self.exploration_factor = Config().server.exploration_factor
        self.exploration_decaying_factor = Config().server.exploration_decaying_factor
        self.min_explore_factor = Config().server.min_explore_factor
        self.explored_clients = []
        self.unexplored_clients = []

        # below are for robustness
        self.robustness = False
        self.augmented_factor = 5
        self.threshold_factor = 1
        self.model_versions_clients_dict = {}
        self.per_round = Config().clients.per_round
        self.reliability_credit_record = {
            client_id: 5 
            for client_id in range(1, self.total_clients + 1)
        }
        self.detected_corrupted_clients = []


    def configure(self):
        """Initialize necessary variables."""
        super().configure()

        self.client_utilities = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
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
            staleness = updates[i].staleness
            staleness_factor = self.staleness_function(staleness)

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples) * staleness_factor

            # Yield to other tasks in the server
            await asyncio.sleep(0)
        return avg_update

    def staleness_function(self, stalenss):
        return 1.0 / pow(stalenss + 1, self.staleness_factor) 

    def weights_aggregated(self, updates):
        """Method called at the end of aggregating received weights."""
        """Calculate client utility here and update the record on the server"""
        for update in updates:
            self.client_utilities[update.client_id] = update.report.statistics_utility * self.staleness_function(update.staleness)
            
            if self.robustness: 
                # Start to do pooling
                start_version = update.report.start_round
                if start_version not in self.model_versions_clients_dict:
                    self.model_versions_clients_dict[start_version] = [(update.client_id, update.report.statistics_utility)]
                else:
                    self.model_versions_clients_dict[start_version].append((update.client_id, update.report.statistics_utility))
                
                tuples = []
                already_existing_clients = set()
                for i in range(self.augmented_factor): 
                    print("start version: ",start_version)
                    print("i: ", i)
                    if start_version - i <= 0:# <0 originally
                        print("should break here")
                        break
                    
                    print("model_versions_clients_dict: ", self.model_versions_clients_dict[start_version-i])

                    tmp = []  # avoid cybil attacks, i.e., outliers repeat deliberately
                    for client_id, loss_norm in self.model_versions_clients_dict[start_version - i]:
                        if client_id in already_existing_clients:
                            continue
                        already_existing_clients.add(client_id)
                        tmp.append((client_id, loss_norm))
                    tuples += tmp#
            
                if len(tuples) >= self.threshold_factor * self.per_round: 
                    logging.info(f"Starting anomaly detection with {len(tuples)} recent records.")
                    self.detect_outliers(tuples)
                else:
                    logging.info(f"Records collected for anomaly detection are not enough: {len(tuples)}.")
        
    def detect_outliers(self, tuples):
     
        client_id_list = [tu[0] for tu in tuples]
        loss_list = [tu[1] for tu in tuples]
        loss_list = np.array(loss_list).reshape(-1, 1)
        min_samples = self.per_round // 2  
        eps = 0.5
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(loss_list)
        result = clustering.labels_.tolist()
        outliers = [client_id_list[idx]
                    for idx, e in enumerate(result) if e == -1]

        newly_detected_outliers = []
        for client_id in outliers:
            self.reliability_credit_record[client_id] -= 1
            if client_id not in self.detected_corrupted_clients:
                current_credit = self.reliability_credit_record[client_id]
                if current_credit == 0:
                    self.detected_corrupted_clients.append(client_id)
                    newly_detected_outliers.append(client_id)

        if len(newly_detected_outliers) == 0:
            #logging.info(f"No new outliers.")
        else:
            newly_detected_outliers = sorted(newly_detected_outliers )
            #logging.info(f"{len(newly_detected_outliers)} clients "
            #             f"are newly taken as outliers"
            #             f": {newly_detected_outliers}.")

    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        selected_clients = []

        if self.robustness:
            outliers = [client_id for client_id in available_clients
                        if client_id in self.detected_corrupted_clients]
            available_clients = [client_id for client_id in available_clients
                                 if client_id not in self.detected_corrupted_clients]
            #logging.info(f"These clients are detected as outliers "
                         #f"and precluded from selection: {outliers}.")

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

            sorted_util = sorted(
                self.client_utilities, key=self.client_utilities.get, reverse=True
            )
        
            selected_clients = sorted_util[:real_exploit_num]
        # Exploration
        random.setstate(self.prng_state)

        # Select unexplored clients randomly
        selected_unexplore_clients = random.sample(
            self.unexplored_clients, clients_count - len(selected_clients))

        self.prng_state = random.getstate()
        self.explored_clients += selected_unexplore_clients

        for client_id in selected_unexplore_clients:
            self.unexplored_clients.remove(client_id)

        selected_clients += selected_unexplore_clients

        logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients
           
