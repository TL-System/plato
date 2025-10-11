"""
Customize the list of inbound and outbound processors through callbacks.
"""

import logging

from dummy_processor import DummyProcessor

from plato.callbacks.client import ClientCallback


class CustomizeProcessorCallback(ClientCallback):
    """
    A client callback that dynamically inserts a dummy processor to the list of inbound processors.
    """

    def on_inbound_process(self, client, inbound_processor):
        """
        Insert a dummy processor to the list of inbound processors.
        """
        logging.info(
            "[%s] Current list of inbound processors: %s.",
            client,
            inbound_processor.processors,
        )
        customized_processor = DummyProcessor(
            client_id=client.client_id,
            current_round=client.current_round,
            name="DummyProcessor",
        )
        inbound_processor.processors.insert(0, customized_processor)

        logging.info(
            "[%s] List of inbound processors after modification: %s.",
            client,
            inbound_processor.processors,
        )

    def on_outbound_process(self, client, outbound_processor):
        """
        Insert a dummy processor to the list of outbound processors.
        """
        logging.info(
            "[%s] Current list of outbound processors: %s.",
            client,
            outbound_processor.processors,
        )
        customized_processor = DummyProcessor(
            client_id=client.client_id,
            current_round=client.current_round,
            name="DummyProcessor",
        )
        outbound_processor.processors.insert(0, customized_processor)

        logging.info(
            "[%s] List of outbound processors after modification: %s.",
            client,
            outbound_processor.processors,
        )
