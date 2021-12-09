"""
Implements a Processor for applying local differential privacy using randomized response.
"""
import logging
from typing import Any

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

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for applying randomized response as the
        local differential privacy mechanism.
        """

        output = super().process(data)

        logging.info(
            "[Client #%d] Local differential privacy (using randomized response) applied.",
            self.client_id)

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:

        if Config().algorithm.epsilon is None:
            return layer

        epsilon = Config().algorithm.epsilon

        layer = layer.detach().cpu().numpy()

        layer = unary_encoding.encode(layer)
        layer = unary_encoding.randomize(layer, epsilon)

        layer = torch.tensor(layer, dtype=torch.float32)

        return layer
