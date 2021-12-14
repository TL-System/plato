"""
Implements a Processor for converting MistNet features from numpy ndarrays to PyTorch tensors.
"""
import logging
from typing import Any

import torch
from plato.processors import base


class Processor(base.Processor):
    """
    Implements a Processor for converting MistNet features from numpy ndarrays to PyTorch tensors.
    """
    def __init__(self, server_id=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.server_id = server_id

    def process(self, data: Any) -> Any:
        """
        Converts MistNet features from numpy ndarrays to PyTorch tensors.
        """
        feature_dataset = []

        for logit, target in data:
            # Uses torch.as_tensor() as opposed to torch.tensor() to avoid data copying
            # according to https://pytorch.org/docs/stable/generated/torch.tensor.html
            feature_dataset.append(
                (torch.as_tensor(logit), torch.as_tensor(target)))

        logging.info(
            "[Server #%d] Features converted from ndarrays to PyTorch tensors.",
            self.server_id)

        return feature_dataset
