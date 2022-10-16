""" 
Customize the inbound and outbound processor list through callbacks. 
"""

import logging
from plato.callbacks.client import ClientCallback

from dummy_processor import DummyProcessor


class CustomizeProcessorCallback(ClientCallback):
    """
    A client callback that dynamically inserts a dummy processor to the existing processor list.
    """

    def on_inbound_process(self, client, inbound_processor):
        """
        Insert a dummy processor to the inbound processor list.
        """
        logging.info(
            "[%s] Current inbound processor list: %s.",
            client,
            inbound_processor.processors,
        )
        customized_processor = DummyProcessor(
            client_id=client.client_id,
            current_round=client.current_round,
            processor_name="DummyProcessor",
        )
        inbound_processor.processors.insert(0, customized_processor)

        logging.info(
            "[%s] Inbound processor list after modification: %s.",
            client,
            inbound_processor.processors,
        )

    def on_outbound_process(self, client, outbound_processor):
        """
        Insert a dummy processor to the outbound processor list.
        """
        logging.info(
            "[%s] Current outbound processor list: %s.",
            client,
            outbound_processor.processors,
        )
        customized_processor = DummyProcessor(
            client_id=client.client_id,
            current_round=client.current_round,
            processor_name="DummyProcessor",
        )
        outbound_processor.processors.insert(0, customized_processor)

        logging.info(
            "[%s] Outbound processor list after modification: %s.",
            client,
            outbound_processor.processors,
        )
