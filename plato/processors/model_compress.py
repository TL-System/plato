"""
Implements a Processor for compressing model weights.
"""
import logging
import pickle
from typing import Any
import zstd

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

        output = zstd.compress(pickle.dumps(data), self.compression_level)

        logging.info("[Client #%d] Compressed model parameters.",
                     self.client_id)

        return output

    def _process_layer(self, layer: Any) -> Any:
        """ No need to process individual layer of the model """
