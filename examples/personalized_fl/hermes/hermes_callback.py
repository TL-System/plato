"""
Callback for attaching a pruning mask to the payload if pruning had been conducted.
"""

import logging

from hermes_processor import SendMaskProcessor

from plato.callbacks.client import ClientCallback


class HermesCallback(ClientCallback):
    """
    A client callback that dynamically inserts processors into the current list of inbound
    processors.
    """

    def on_outbound_ready(self, client, report, outbound_processor):
        """
        Insert a SendMaskProcessor to the list of outbound processors.
        """
        send_payload_processor = SendMaskProcessor(
            client_id=client.client_id,
            name="SendMaskProcessor",
        )

        outbound_processor.processors.insert(0, send_payload_processor)

        logging.info(
            "[%s] List of outbound processors: %s.",
            client,
            outbound_processor.processors,
        )
