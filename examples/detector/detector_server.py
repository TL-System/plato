"""
A customized server with detector so that poisoned model updates can be filtered out.
"""

import logging

from plato.config import Config
from plato.servers import fedavg
from collections import OrderedDict
import attacks as attack_registry
import defences as defence_registry
import aggregations as aggregation_registry

import numpy as np
import torch
import defences

from typing import Mapping
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

    def configure(self):
        """Initialize defence related parameter"""
        super().configure()

        self.attacker_list = [
            int(value) for value in Config().clients.attacker_ids.split(",")
        ]
        self.attack_type = Config().clients.attack_type

        logging.info(f"self.attacker_ids: %s", self.attacker_list)
        logging.info(f"attack_type: %s", self.attack_type)

    def choose_clients(self, clients_pool, clients_count):
        selected_clients = super().choose_clients(clients_pool, clients_count)

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

        # Attacker server perform attack based on attack type
        attack = attack_registry.get()
        weights_attacked = attack(
            attacker_weights
        )  # weights and updates are different, think about which is more convenient?

        # Put poisoned model back to weights received for further aggregation
        counter = 0
        for i, update in enumerate(self.updates):
            if update.client_id in self.attacker_list:
                weights_received[i] = weights_attacked[counter]
                counter += 1

        return weights_received

    def weights_filter(self, weights_attacked):

        # Identify poisoned updates and remove it from all received updates.
        defence = defence_registry.get()

        weights_approved = defence(weights_attacked)
        # get a balck list for attackers_detected this round

        # Remove identified attacker from clients pool. Never select that client again.
        # for attacker in attackers_detected:
        #    self.clients_pool.remove(attacker)

        return weights_approved

    async def aggregate_weights(self, updates,baseline_weights, weights_received):
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
