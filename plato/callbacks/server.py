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

    def on_client_arrived(self, server, **kwargs):
        """
        Event called after a new client arrived.
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
        logging.info("[%s] Updated weights have been received.", server)

    def on_weights_aggregated(self, server, updates):
        """
        Event called after the updated weights have been aggregated.
        """
        logging.info("[%s] Finished aggregating updated weights.", server)

    def on_clients_selected(self, server, selected_clients):
        """
        Event called after clients have been selected in each round.
        """

    def on_clients_processed(self, server):
        """
        Event called after all client reports have been processed.
        """
        logging.info("[%s] All client reports have been processed.", server)

    def on_training_will_start(self, server):
        """
        Event called before selecting clients for the first round of training.
        """
        logging.info("[%s] Starting training.", server)

    def on_server_will_close(self, server):
        """
        Event called at the start of closing the server.
        """
        logging.info("[%s] Closing the server.", server)
