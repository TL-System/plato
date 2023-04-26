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

    def configure(self) -> None:
        """Initialize the attack related parameter"""
        return super().configure()
