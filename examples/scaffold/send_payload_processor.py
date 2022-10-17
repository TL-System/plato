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

        client_control_variate_filename = self.trainer.client_control_variate_path
        if os.path.exists(client_control_variate_filename):
            with open(client_control_variate_filename, "rb") as payload_file:
                client_control_variate = pickle.load(payload_file)
                data = [data, client_control_variate]
        else:
            data = [data, None]

        if data[1] is not None:
            logging.info(
                "[Client #%d] Extra information attached to payload.",
                self.client_id,
            )
        return data
