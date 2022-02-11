"""
Implements a Processor for decompressing model weights.
"""

import logging
from typing import Any
import pickle
import zstd

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor for decompressing of model parameters.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, data: Any) -> Any:
        """ Implements a Processor for decompressing model parameters. """

        output = pickle.loads(zstd.decompress(data))

        logging.info("[Server #%d] Decompressed received model parameters.",
                     self.server_id)

        return output

    def _process_layer(self, layer: Any) -> Any:
        """ No need to process individual layer of the model """
