"""
Defines the TrainerCallback class, which is the abstract base class to be subclassed
when creating new client callbacks.

Defines a default callback to print local training progress.
"""
import logging
from abc import ABC

from plato.utils import fonts


class ClientCallback(ABC):
    """
    The abstract base class to be subclassed when creating new client callbacks.
    """

    def on_client_train_end(self, client, **kwargs):
        """
        Event called at the end of local training.
        """


class PrintProgressCallback(ClientCallback):
    """
    A callback which prints a message when needed.
    """

    def on_client_train_end(self, client, **kwargs):
        """
        Event called at the end of local training.
        """
