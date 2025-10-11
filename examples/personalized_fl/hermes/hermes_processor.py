"""
An outbound processor for Hermes to load a mask from the local file system on the client,
and attach it to the payload.
"""

import logging
import os
import pickle
from typing import OrderedDict

from plato.config import Config
from plato.processors import base


class SendMaskProcessor(base.Processor):
    """
    Implements a processor for attaching a pruning mask to the payload if pruning
    had been conducted.
    """

    def __init__(self, client_id, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id

    def process(self, data: OrderedDict):
        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        model_path = Config().params["model_path"]

        mask_filename = f"{model_path}/{model_name}_client{self.client_id}_mask.pth"
        if os.path.exists(mask_filename):
            with open(mask_filename, "rb") as payload_file:
                client_mask = pickle.load(payload_file)
                data = [data, client_mask]
        else:
            data = [data, None]

        if data[1] is not None:
            if self.client_id is None:
                logging.info(
                    "[Server #%d] Pruning mask attached to payload.", self.server_id
                )
            else:
                logging.info(
                    "[Client #%d] Pruning mask attached to payload.", self.client_id
                )
        return data
