"""
Processor for attaching a pruning mask to the payload if pruning
had been conducted
"""

import logging
from typing import OrderedDict
import torch
from plato.processors import model


class Processor(model.Processor):
    """
    Implements a processor for attaching a pruning mask to the payload if pruning
    had been conducted
    """

    def process(self, data: OrderedDict) -> OrderedDict:

        data = [data, self.trainer.mask]
        if self.client_id is None:
            logging.info(
                "[Server #%d] Attached pruning mask to payload.", self.server_id
            )
        else:
            logging.info(
                "[Client #%d] Attached pruning mask to payload.", self.client_id
            )
        return data

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """
        Process individual layer of the model
        """
