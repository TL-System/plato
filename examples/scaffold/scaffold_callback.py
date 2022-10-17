""" 
Customize the inbound and outbound processor list for scaffold clients through callbacks. 
"""

import logging
from plato.callbacks.client import ClientCallback

from extract_payload_processor import ExtractPayloadProcessor
from send_payload_processor import SendPayloadProcessor


class CustomizeProcessorCallback(ClientCallback):
    """
    A client callback that dynamically inserts a dummy processor to the existing processor list.
    """

    def on_inbound_process(self, client, inbound_processor):
        """
        Insert an ExtractPayloadProcessor to the inbound processor list.
        """
        logging.info(
            "[%s] Current list of inbound processors: %s.",
            client,
            inbound_processor.processors,
        )
        customized_processor = ExtractPayloadProcessor(
            client_id=client.client_id,
            current_round=client.current_round,
            name="ExtractPayloadProcessor",
        )
        inbound_processor.processors.insert(0, customized_processor)

        logging.info(
            "[%s] List of inbound processors after modification: %s.",
            client,
            inbound_processor.processors,
        )

    def on_outbound_process(self, client, outbound_processor):
        """ "
        Insert a SendPayloadProcessor to the outbound processor list.
        """
        logging.info(
            "[%s] Current list of outbound processors: %s.",
            client,
            outbound_processor.processors,
        )
        customized_processor = SendPayloadProcessor(
            client_id=client.client_id,
            current_round=client.current_round,
            name="SendPayloadProcessor",
        )
        outbound_processor.processors.insert(0, customized_processor)

        logging.info(
            "[%s] List of inbound processors after modification: %s.",
            client,
            outbound_processor.processors,
        )
