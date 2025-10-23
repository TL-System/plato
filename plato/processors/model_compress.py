"""
Implements a Processor for compressing model weights.
"""

import logging
import pickle
from typing import Any

from plato.processors import model
from plato.utils.zstd_helpers import get_zstd

zstd = get_zstd()


class Processor(model.Processor):
    """
    Implements a Processor for compressing model parameters.
    """

    def __init__(self, compression_level=1, **kwargs) -> None:
        super().__init__(**kwargs)

        self.compression_level = compression_level

    def process(self, data: Any) -> Any:
        """Implements a Processor for compressing model parameters."""

        output = zstd.compress(pickle.dumps(data), self.compression_level)

        if self.client_id is None:
            logging.info("[Server #%d] Compressed model parameters.", self.server_id)
        else:
            logging.info("[Client #%d] Compressed model parameters.", self.client_id)

        return output
