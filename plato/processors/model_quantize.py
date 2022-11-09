"""
Implements a Processor for quantizing model parameters.
"""

import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor to quantize model parameters to 16-bit floating points.
    """

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """Dequantizes each individual layer of the model."""

        return layer.to(torch.bfloat16)
