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
    def __init__(self,
                 scale=0.1,
                 zero_point=10,
                 dtype=torch.quint8,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for quantizing model parameters.
        """

        output = super().process(data)

        logging.info("[Client #%d] Quantization applied.", self.client_id)

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:

        layer = torch.quantize_per_tensor(layer, self.scale, self.zero_point,
                                          self.dtype)

        return layer
