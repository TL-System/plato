"""
Processor for attaching a pruning mask to the payload if pruning
had been conducted
"""

import os
import pickle
import logging
from typing import OrderedDict
import torch
from plato.processors import model
from plato.config import Config


class Processor(model.Processor):
    """
    Implements a processor for attaching a pruning mask to the payload if pruning
    had been conducted
    """

    def process(self, data: OrderedDict):
        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        checkpoint_path = Config().params["checkpoint_path"]

        mask_filename = (
            f"{checkpoint_path}/{model_name}_client{self.client_id}_mask.pth"
        )
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

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """
        Process individual layer of the model
        """
