"""
Implements a Processor for compressing model weights.
"""
import logging
from typing import Any
import zstd
import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for compressing of model parameters.
    """
    def __init__(self, compression_level=1, **kwargs) -> None:
        super().__init__(**kwargs)

        self.compression_level = compression_level

    def process(self, data: Any) -> Any:
        """ Implements a Processor for compressing model parameters. """

        output = super().process(data)

        logging.info("[Client #%d] Compressed model parameters.",
                     self.client_id)

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """ Compress individual layer of the model. """

        data = layer.detach().cpu().numpy()

        compressed_layer = (data.shape, data.dtype,
                            zstd.compress(data, self.compression_level))

        return compressed_layer
