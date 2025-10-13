"""
Obtaining a model from the PyTorch Hub.
"""

import torch


class Model:
    """
    The model loaded from PyTorch Hub.

    We will soon be using the get_model() method for torchvision 0.14 when it is released.
    """

    @staticmethod
    # pylint: disable=unused-argument
    def get(model_name=None, **kwargs):
        """Returns a named model from PyTorch Hub.

        Note: For torchvision models, use 'weights' parameter instead of deprecated 'pretrained'.
        For backward compatibility, 'pretrained=True' is automatically converted to
        'weights="DEFAULT"' and 'pretrained=False' to 'weights=None'.
        """
        # Convert deprecated 'pretrained' parameter to 'weights' for torchvision compatibility
        if "pretrained" in kwargs:
            pretrained = kwargs.pop("pretrained")
            if "weights" not in kwargs:
                kwargs["weights"] = "DEFAULT" if pretrained else None

        return torch.hub.load("pytorch/vision", model_name, **kwargs)
