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

    def on_client_train_start(self, client, **kwargs):
        """
        Event called at the start of local training.
        """

    def on_client_train_end(self, client, **kwargs):
        """
        Event called at the end of local training.
        """

    def on_client_send_with_comm_simulation_end(self, client, data_size, **kwargs):
        """
        Event called at the end of sending payload using simulation.
        """

    def on_client_send_without_comm_simulation_end(self, client, data_size, **kwargs):
        """
        Event called at the end of sending payload using S3 or socket.io.
        """


class PrintProgressCallback(ClientCallback):
    """
    A callback which prints a message when needed.
    """

    def on_client_train_start(self, client, **kwargs):
        """
        Event called at the start of local training.
        """
        logging.info(
            fonts.colourize(
                f"[{client}] Started training in communication round #{client.current_round}."
            )
        )

    def on_client_send_with_comm_simulation_end(self, client, data_size, **kwargs):
        """
        Event called at the end of sending payload using simulation.
        """
        logging.info(
            "[%s] Sent %.2f MB of payload data to the server (simulated).",
            client,
            data_size / 1024**2,
        )

    def on_client_send_without_comm_simulation_end(self, client, data_size, **kwargs):
        """
        Event called at the end of sending payload using S3 or socket.io.
        """
        logging.info(
            "[%s] Sent %.2f MB of payload data to the server.",
            client,
            data_size / 1024**2,
        )
