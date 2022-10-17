"""
Customize the list of inbound and outbound processors for scaffold clients through callbacks.
"""

import logging

from plato.callbacks.client import ClientCallback

from extract_payload_processor import ExtractPayloadProcessor
from send_payload_processor import SendPayloadProcessor


class ScaffoldProcessorCallback(ClientCallback):
    """
    A client callback that dynamically inserts processors into the current list of inbound processors.
    """

    def on_inbound_process(self, client, inbound_processor):
        """
        Insert an ExtractPayloadProcessor to the list of inbound processors.
        """
        extract_payload_processor = ExtractPayloadProcessor(
            client_id=client.client_id,
            trainer=client.trainer,
            name="ExtractPayloadProcessor",
        )
        inbound_processor.processors.insert(0, extract_payload_processor)

        logging.info(
            "[%s] List of inbound processors: %s.", client, inbound_processor.processors
        )

    def on_outbound_process(self, client, outbound_processor):
        """
        Insert a SendPayloadProcessor to the list of outbound processors.
        """
        send_payload_processor = SendPayloadProcessor(
            client_id=client.client_id,
            trainer=client.trainer,
            name="SendPayloadProcessor",
        )

        outbound_processor.processors.insert(0, send_payload_processor)

        logging.info(
            "[%s] List of outbound processors: %s.",
            client,
            outbound_processor.processors,
        )
