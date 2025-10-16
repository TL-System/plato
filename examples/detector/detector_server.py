"""
A customized server with detector so that poisoned model updates can be filtered out.
"""

import csv
import logging
import os
from collections import OrderedDict
from typing import Mapping

import aggregations as aggregation_registry
import attacks as attack_registry
import defences
import detectors as defence_registry
import numpy as np
import torch

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
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
        self.attacker_list = None
        self.attack_type = None
        self.blacklist = []
        self.pre_blacklist = []

    def configure(self):
        """Initialize defence related parameter"""
        super().configure()

        self.attacker_list = [
            int(value) for value in Config().clients.attacker_ids.split(",")
        ]
        self.attack_type = (
            Config().clients.attack_type
            if hasattr(Config().clients, "attack_type")
            else None
        )

        logging.info(f"self.attacker_ids: %s", self.attacker_list)
        logging.info(f"attack_type: %s", self.attack_type)

    def choose_clients(self, clients_pool, clients_count):
        # remove clients in blacklist from available clients pool
        # logging.info(f"len of clients pool before removal: %d", len(clients_pool))
        clients_pool = list(filter(lambda x: x not in self.blacklist, clients_pool))
        # logging.info(f"len of cliets pool after removal: %d", len(clients_pool))

        selected_clients = self._select_clients_with_strategy(
            clients_pool, clients_count
        )

        # recording how many attackers are selected this round to track the defence performance
        selected_attackers = []
        for select_client in selected_clients:
            if select_client in self.attacker_list:
                selected_attackers.append(select_client)

        logging.info("[%s] Selected attackers: %s", self, selected_attackers)

        return selected_clients

    def weights_received(self, weights_received):
        """
        Attacker server performs attack based on malicious clients' reports and aggregation server defences attacks.
        """
        # Simulate the attacker server to perform model poisoning. Note that the attack server only accesses to malicious clients' updates.
        weights_attacked = self.weights_attacked(weights_received)

        # Simulate the aggregation server to filter out poisoned reports before performing aggregation.
        weights_approved = self.weights_filter(weights_attacked)

        return weights_approved

    def weights_attacked(self, weights_received):
        # Extract attackers' updates
        attacker_weights = []
        for weight, update in zip(weights_received, self.updates):
            if update.client_id in self.attacker_list:
                attacker_weights.append(weight)

        # Extract model updates
        baseline_weights = self.algorithm.extract_weights()
        deltas_received = self.algorithm.compute_weight_deltas(
            baseline_weights, attacker_weights
        )
        # Get attackers selected at this round
        received_ids = [update.client_id for update in self.updates]
        num_attackers = len([i for i in received_ids if i in self.attacker_list])

        if num_attackers > 0:
            # Attacker server perform attack based on attack type
            attack = attack_registry.get()
            weights_attacked = attack(
                baseline_weights, attacker_weights, deltas_received, num_attackers
            )  # weights and updates are different, think about which is more convenient?

            # Put poisoned model back to weights received for further aggregation
            counter = 0
            for i, update in enumerate(self.updates):
                if update.client_id in self.attacker_list:
                    weights_received[i] = weights_attacked[counter]
                    counter += 1

        return weights_received

    def detect_analysis(self, detected_malicious_ids, received_ids):
        "print out detect accuracy, positive rate and negative rate"
        logging.info(f"detected ids: %s", detected_malicious_ids)
        real_malicious_ids = [i for i in received_ids if i in self.attacker_list]
        logging.info(f"real attackers id: %s", real_malicious_ids)
        if len(real_malicious_ids) != 0:
            correct = 0
            wrong = 0
            for i in detected_malicious_ids:
                if i in real_malicious_ids:
                    correct += 1
                    logging.info(f"correctly detectes attacker %d", i)
                else:
                    wrong += 1
                    logging.info(f"wrongly classify benign client %i into attacker", i)
            detection_accuracy = correct / (len(real_malicious_ids) * 1.0)
            with open("detection_accuracy.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([detection_accuracy])
            logging.info(f"detection_accuracy is: %.2f", detection_accuracy)
            logging.info(
                f"Missing %d attackers.", len(real_malicious_ids) * 1.0 - correct
            )
            logging.info(f"falsely classified %d clients: ", wrong)

    def weights_filter(self, weights_attacked):
        # Identify poisoned updates and remove it from all received updates.
        defence = defence_registry.get()
        if defence is None:
            return weights_attacked

        # Extract the current model updates (deltas)
        baseline_weights = self.algorithm.extract_weights()
        deltas_attacked = self.algorithm.compute_weight_deltas(
            baseline_weights, weights_attacked
        )
        received_ids = [update.client_id for update in self.updates]
        received_staleness = [update.staleness for update in self.updates]
        malicious_ids, weights_approved = defence(
            baseline_weights,
            weights_attacked,
            deltas_attacked,
            received_ids,
            received_staleness,
        )

        ids = [received_ids[i] for i in malicious_ids]

        cummulative_detect = 0
        for id_temp in self.blacklist:
            if id_temp in self.attacker_list:
                cummulative_detect += 1
                # logging.info(f"cummulative detect: %d",cummulative_detect)

        # logging.info(f"Cumulative detection: %.2f", (cummulative_detect) * 1.0 / len(self.attacker_list))
        # logging.info(f"Mistakenly classfied: %d benign clients so far.", (len(self.blacklist)-cummulative_detect))
        # logging.info(f"Blacklist is: %s",  self.blacklist)
        """
        self.blacklist[name].append()
        # Remove identified attacker from client pool. Never select that client again.
        for i in ids: 
            self.clients_pool.remove(i)
            logging.info(f"Remove attacker %d from available client pool.", i)
        """
        # Analyze detection performance.
        # self.detect_analysis(ids, received_ids)

        return weights_approved

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate the reported weight updates from the selected clients."""

        if not hasattr(Config().server, "secure_aggregation_type"):
            logging.info(f"Fedavg is applied.")
            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights, weights_received
            )
            deltas = await self.aggregate_deltas(self.updates, deltas_received)
            updated_weights = self.algorithm.update_weights(deltas)
            return updated_weights

        # if secure aggregation is applied.
        aggregation = aggregation_registry.get()

        weights_aggregated = aggregation(updates, baseline_weights, weights_received)

        return weights_aggregated
