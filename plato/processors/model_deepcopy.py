"""
Processor for creating a deep copy of the PyTorch model state_dict.
"""

import copy

import torch
from plato.processors import model


class Processor(model.Processor):
    """
    Processor for creating a deep copy of the PyTorch model state_dict.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _process_layer(self, layer: torch.Tensor):
        return copy.deepcopy(layer)
