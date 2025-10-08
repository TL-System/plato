"""
Implements a Processor for applying local differential privacy using randomized response.
"""

import torch

from plato.config import Config
from plato.processors import model
from plato.utils import unary_encoding


class Processor(model.Processor):
    """
    Implements a Processor for applying local differential privacy using randomized response.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        if Config().algorithm.epsilon is None:
            return layer

        epsilon = Config().algorithm.epsilon

        # Apply randomized response as the local differential privacy mechanism
        layer = layer.detach().cpu().numpy()

        layer = unary_encoding.encode(layer)
        layer = unary_encoding.randomize(layer, epsilon)

        layer = torch.tensor(layer, dtype=torch.float32)

        return layer
