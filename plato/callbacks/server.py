"""
Defines the ServerCallback class, which is the abstract base class to be subclassed
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

    def on_weights_received(self, server, weights_received):
        """
        Event called after the updated weights have been received.
        """

    def on_weights_aggregated(self, server, updates):
        """
        Event called after the updated weights have been aggregated.
        """

    def on_server_will_close(self, server, **kwargs):
        """
        Event called at the start of closing the server.
        """


class PrintProgressCallback(ServerCallback):
    """
    A callback which prints a message when needed.
    """

    def on_weights_received(self, server, weights_received):
        """
        Event called after the updated weights have been received.
        """
        logging.info("[Server #%s] Updated weights have been received.", os.getpid())

    def on_weights_aggregated(self, server, updates):
        """
        Event called after the updated weights have been aggregated.
        """
        logging.info("[Server #%s] Finished aggregating updated weights.", os.getpid())

    def on_server_will_close(self, server, **kwargs):
        """
        Event called at the start of closing the server.
        """
        logging.info("[Server #%s] Closing the server.", os.getpid())
