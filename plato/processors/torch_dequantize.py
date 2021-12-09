"""
Implements a Processor for dequantizing model parameters.
"""
import logging
from typing import Any

import torch

from plato.processors import torch_model


class Processor(torch_model.Processor):
    """
    Implements a Processor for dequantizing model parameters.
    """
    def __init__(self, server_id=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.server_id = server_id

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
