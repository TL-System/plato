"""
Implements a Processor for quantizing model parameters.
"""
import logging
from typing import Any

import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for quantizing model parameters.
    """

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for quantizing model parameters.
        """

        output = super().process(data)

        if self.client_id is None:
            logging.info(
                "[Server #%d] Quantized the model to 16-bit floating points.",
                self.server_id)
        else:
            logging.info(
                "[Client #%d] Quantized the update to 16-bit floating points.",
                self.client_id)

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:

        return layer.to(torch.bfloat16)
