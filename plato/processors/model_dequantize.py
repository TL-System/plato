"""
Implements a Processor for dequantizing model parameters.
"""
import logging
from typing import Any

import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for dequantizing model parameters.
    """

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for dequantizing model parameters.
        """

        output = super().process(data)

        if self.client_id is None:
            logging.info("[Server #%d] Dequantized received update.",
                         self.server_id)
        else:
            logging.info("[Client #%d] Dequantized received global model.",
                         self.client_id)

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:

        return layer.to(torch.float32)
