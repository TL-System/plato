"""
An outbound processor for Hermes to load a mask from the local file system on the client,
and attach it to the payload.
"""

import logging
import os
import pickle
from collections import OrderedDict
from typing import Any, List, Optional

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

    def process(self, data: OrderedDict[Any, Any]) -> List[Any]:
        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        model_path = Config().params["model_path"]

        mask_filename = f"{model_path}/{model_name}_client{self.client_id}_mask.pth"
        client_mask: Optional[OrderedDict[Any, Any]] = None
        if os.path.exists(mask_filename):
            with open(mask_filename, "rb") as payload_file:
                client_mask = pickle.load(payload_file)

        payload_with_mask: List[Any] = [data, client_mask]

        if payload_with_mask[1] is not None:
            if self.client_id is None:
                server_id = getattr(self, "server_id", None)
                if server_id is not None:
                    logging.info(
                        "[Server #%d] Pruning mask attached to payload.", server_id
                    )
                else:
                    logging.info("[Server] Pruning mask attached to payload.")
            else:
                logging.info(
                    "[Client #%d] Pruning mask attached to payload.", self.client_id
                )
        return payload_with_mask
