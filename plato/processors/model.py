"""
Base processor for processing PyTorch models.
"""

import logging
import pickle
import sys
from typing import OrderedDict

import torch

from plato.processors import base


class Processor(base.Processor):
    """Base processor for processing PyTorch models."""

    def __init__(self, client_id=None, server_id=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client_id = client_id
        self.server_id = server_id

    def process(self, data: OrderedDict) -> OrderedDict:
        """
        Processes PyTorch model parameter.
        The data is a state_dict of a PyTorch model.
        """
        old_data_size = sys.getsizeof(pickle.dumps(data))

        new_data = OrderedDict()
        for layer_name, layer_params in data.items():
            new_data[layer_name] = self._process_layer(layer_params)

        new_data_size = sys.getsizeof(pickle.dumps(new_data))

        if self.client_id is None:
            logging.info(
                "[Server #%d] Processed the model and changed its size from %.2f MB to %.2f MB.",
                self.server_id,
                old_data_size / 1024**2,
                new_data_size / 1024**2,
            )
        else:
            logging.info(
                "[Client #%d] Processed the model and changed its size from %.2f MB to %.2f MB.",
                self.client_id,
                old_data_size / 1024**2,
                new_data_size / 1024**2,
            )

        return new_data

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """Processes an individual layer of the model."""
        return layer
