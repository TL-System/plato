"""
Implements a Processor for dequantizing model parameters.
"""

import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for dequantizing model parameters.
    """

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """Quantizes each individual layer of the model."""

        return layer.to(torch.float32)
