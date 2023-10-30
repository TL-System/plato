"""
The processors for loading, assigning, and adjusting the models for local update
of clients.
"""

import logging
from typing import Any

from plato.processors import base
from plato.callbacks import client as client_callbacks


class PayloadStatusProcessor(base.Processor):
    """
    A default payload status processor to present what modules are
    received contained in the payload.
    """

    def __init__(self, algorithm, **kwargs) -> None:
        super().__init__(**kwargs)
        self.algorithm = algorithm

    def process(self, data: Any) -> Any:
        if isinstance(data, (list, tuple)):
            payload = data[0]
        else:
            payload = data
        logging.info(
            "[Client #%d] Received the payload containing modules: %s.",
            self.trainer.client_id,
            self.algorithm.extract_modules_name(list(payload.keys())),
        )

        return data


class ClientPayloadCallback(client_callbacks.ClientCallback):
    """
    A default client callback to process the received payload.
    """

    def on_inbound_received(self, client, inbound_processor):
        inbound_processor.processors.append(
            PayloadStatusProcessor(trainer=client.trainer, algorithm=client.algorithm)
        )
