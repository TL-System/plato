"""
Base processor for processing PyTorch models.
"""

from abc import abstractmethod
from typing import OrderedDict

import torch
from plato.processors import base


class Processor(base.Processor):
    """Base processor for processing PyTorch models."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def process(self, data: OrderedDict) -> OrderedDict:
        """
        Processing PyTorch model parameter.
        The data is a state_dict of a PyTorch model.
        """
        new_data = OrderedDict()
        for layer_name, layer_params in data.items():
            new_data[layer_name] = self._process_layer(layer_params)

        return new_data

    @abstractmethod
    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """
        Process individual layer of the model
        """
