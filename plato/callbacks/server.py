"""
Defines the TrainerCallback class, which is the abstract base class to be subclassed
when creating new server callbacks.

Defines a default callback to print training progress.
"""
import logging
import os
from abc import ABC


class ServerCallback(ABC):
    """
    The abstract base class to be subclassed when creating new server callbacks.
    """

    def on_server_selection_start(self, server, **kwargs):
        """
        Event called at the start of selecting clients.
        """

    def on_server_selection_end(self, server, **kwargs):
        """
        Event called at the end of selecting clients.
        """

    def on_server_close_start(self, server, **kwargs):
        """
        Event called at the start of closing the server.
        """


class PrintProgressCallback(ServerCallback):
    """
    A callback which prints a message at the start of each epoch, and at the end of each step.
    """

    def on_server_selection_start(self, server, **kwargs):
        """
        Event called at the start of selecting clients.
        """
        logging.info("[Server #%s] Selecting clients.", os.getpid())

    def on_server_close_start(self, server, **kwargs):
        """
        Event called at the start of closing the server.
        """
        logging.info("[Server #%s] Closing the server.", os.getpid())
