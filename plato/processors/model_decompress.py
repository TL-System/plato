"""
Implements a Processor for decompressing model weights.
"""

import logging
from typing import Any
import numpy as np
import zstd
import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for decompressing of model parameters.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, data: Any) -> Any:
        """ Implements a Processor for decompressing model parameters. """

        output = super().process(data)

        logging.info("[Server #%d] Decompressed received model parameters.",
                     self.server_id)

        return output

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """ Compress individual layer of the model. """

        shape, dtype, modelcom = layer
        modelcom = zstd.decompress(modelcom)
        decompressed_layer = np.fromstring(modelcom, dtype).reshape(shape)

        return torch.from_numpy(decompressed_layer)
