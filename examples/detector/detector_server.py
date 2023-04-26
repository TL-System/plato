"""
A customized server with detector so that poisoned model updates can be filtered out.
"""

import logging

from plato.config import Config
from plato.servers import fedavg
from collections import OrderedDict
import attacks as attack_registry
import defences as defence_registry

import numpy as np
import torch
import defences


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
        super.configure()

        self.attacker_list = Config().clients.attacker_ids
        self.attack_type = Config().clients.attack_type

    def weights_received(self, updates):
        """
        Attacker server performs attack based on malicious clients' reports and aggregation server defences attacks.
        """
        # Simulate the attacker server to perform model poisoning. Note that the attack server only accesses to malicious clients' updates.
        weights_attacked = self.weights_attacked(updates)

        # Simulate the aggregation server to filter out poisoned reports before performing aggregation.
        weights_approved = self.weights_filter(weights_attacked)

        return weights_approved

    def weights_attacked(self, updates):
        # Extract attackers' updates
        attacker_updates = []
        for update in updates:
            if update.client_id in self.attacker_list:
                attacker_updates.append(update)

        # Attacker server perform attack based on attack type
        attack = attack_registry.get()
        weights_attacked = attack(attacker_updates)

        return weights_attacked

    def weights_filter(self, weights_attacked):

        # Identify poisoned updates and remove it from all received updates.
        defence = defence_registry.get()

        weights_approved, attackers_detected = defence(weights_attacked)

        # Remove identified attacker from clients pool. Never select that client again.
        for attacker in attackers_detected:
            self.clients_pool.remove(attacker)

        return weights_approved
