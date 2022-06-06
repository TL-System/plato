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
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for dequantizing model parameters.
        """

        output = super().process(data)

        logging.info("[Server #%d] Dequantized features.", self.server_id)

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:

        layer = torch.dequantize(layer)

        return layer
