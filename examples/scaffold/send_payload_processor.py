"""
Processor for attaching additional items to the payload sent by the client.
"""

import os
import pickle
import logging
from typing import Any, List

from plato.processors import base


class SendPayloadProcessor(base.Processor):
    """
    A processor for clients to attach additional items to the client payload.
    """

    def __init__(self, client_id, trainer, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id
        self.trainer = trainer

    def process(self, data: Any) -> List:

        extra_payload_filename = self.trainer.extra_payload_path
        if os.path.exists(extra_payload_filename):
            with open(extra_payload_filename, "rb") as payload_file:
                extra_payload = pickle.load(payload_file)
                data = [data, extra_payload]
        else:
            data = [data, None]

        if data[1] is not None:
            logging.info(
                "[Client #%d] Extra information attached to payload.",
                self.client_id,
            )
        return data
