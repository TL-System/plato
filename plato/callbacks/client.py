"""
Defines the TrainerCallback class, which is the abstract base class to be subclassed
when creating new client callbacks.

Defines a default callback to print local training progress.
"""

from abc import ABC
import logging


class ClientCallback(ABC):
    """
    The abstract base class to be subclassed when creating new client callbacks.
    """

    def on_inbound_payload_received(self, client, inbound_processor):
        """
        Event called before inbound processors start to process data.
        """

    def on_inbound_payload_processed(self, client, processed_payload):
        """
        Event called when payload was processed by inbound processors.
        """

    def on_outbound_payload_ready(self, client, outbound_processor):
        """
        Event called before outbound processors start to process data.
        """


class LogProgressCallback(ClientCallback):
    """
    A callback which prints a message when needed.
    """

    def on_inbound_payload_received(self, client, inbound_processor):
        """
        Event called before inbound processors start to process data.
        """
        logging.info("[%s] Start to process inbound data.", client)

    def on_inbound_payload_processed(self, client, processed_payload):
        """
        Event called when payload was processed by inbound processors.
        """
        logging.info("[%s] Inbound data has been processed.", client)

    def on_outbound_payload_ready(self, client, outbound_processor):
        """
        Event called before outbound processors start to process data.
        """
        logging.info("[%s] Start to process outbound data.", client)
