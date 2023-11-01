"""
The processors for processing the received payload.
"""

import logging
from typing import Any

from plato.processors import base
from plato.callbacks import client as client_callbacks
from pflbases.fedavg_partial import Algorithm


class PayloadStatusProcessor(base.Processor):
    """
    A default payload status processor to present what layers are
    contained in the received payload.
    """

    def process(self, data: Any) -> Any:
        if isinstance(data, (list, tuple)):
            payload = data[0]
        else:
            payload = data
        logging.info(
            "[Client #%d] Received the payload containing layers: %s.",
            self.trainer.client_id,
            Algorithm.extract_layer_names(list(payload.keys())),
        )

        return data


class ClientPayloadCallback(client_callbacks.ClientCallback):
    """
    A default client callback to process the received payload.
    """

    def on_inbound_received(self, client, inbound_processor):
        inbound_processor.processors.append(
            PayloadStatusProcessor(trainer=client.trainer)
        )
