"""
A federated learning client that is capable to perform model poisoning attacks.
"""

import logging
import os
import pickle

from plato.clients import simple
from plato.config import Config
from plato.utils import fonts

import attacks


class Client(simple.Client):
    """A client who is able to perform model poisoning attack"""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
    ):
        super().__init__(
            model, datasource, algorithm, trainer, callbacks, trainer_callbacks
        )
        self.is_attacker = None
        self.attack_type = None

    def configure(self) -> None:
        """Initialize the attack related parameter"""
        super().configure()

        self.is_attacker = self.client_id in Config().clients.attacker_ids
        self.attack_type = Config().clients.attack_type


