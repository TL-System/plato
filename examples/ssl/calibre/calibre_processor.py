"""
An outbound prossor for Calibre algorithm to save the divergence on the client locally.
"""
import os
import logging
from typing import OrderedDict

import torch

from plato.processors import base
from plato.config import Config


class AddDivergenceRateProcessor(base.Processor):
    """
    Implement a processor for adding the divergence rate to the payload.
    """

    def __init__(self, client_id, trainer, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id
        self.trainer = trainer

    def process(self, data: OrderedDict):
        """Process the payload by adding the computed divergence rate to the payload."""
        model_path = Config().params["model_path"]
        filename = f"client_{self.client_id}_divergence_rate.pth"
        save_path = os.path.join(model_path, filename)

        divergence_rate = torch.load(save_path)

        data = [data, divergence_rate]

        logging.info(
            "[Client #%d] Divergence Rate attached to payload.", self.client_id
        )
        return data
