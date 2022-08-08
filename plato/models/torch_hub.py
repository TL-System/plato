"""
Obtaining a model from the PyTorch Hub.
"""

import torch


class Model:
    """The model loaded from PyTorch Hub."""

    @staticmethod
    def get(model_name=None, num_classes=None, pretrained=False):
        weights = None
        if pretrained:
            weights = None
        return torch.hub.load("pytorch/vision", model_name, weights=weights)
