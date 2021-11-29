"""
Processor for converting features in ndarray into pytorch tensors.
Only used for features in mistnet in pytorch.
"""
from typing import Any
import logging
import os

import torch

from plato.processors import base


class Processor(base.Processor):
    """
    Processor class.
    Processor for converting features in ndarray into pytorch tensors.
    """
    def __init__(self, *args, server_id=None, **kwargs) -> None:
        """Constructor for Processor"""
        self.server_id = server_id
        if server_id is None:
            self.server_id = os.getpid()

    def process(self, data: Any) -> Any:
        """
        Data processing implementation.
        Converting features in ndarray into pytorch tensors.
        """

        feature_dataset = []

        for logit, target in data:
            feature_dataset.append((torch.tensor(logit), torch.tensor(target)))

        logging.info("[Server #%d] Features converted from ndarray to tensor.",
                     self.server_id)

        return feature_dataset
