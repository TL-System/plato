"""
Implements a Processor for applying dequantization to MistNet PyTorch features.
"""
import logging
from typing import Any

import torch
from plato.processors import base


class Processor(base.Processor):
    """
    Implements a Processor for applying dequantization to MistNet PyTorch features.
    """
    def __init__(self, server_id=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.server_id = server_id

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for applying dequantization to MistNet PyTorch features.
        """
        feature_dataset = []

        for logit, target in data:
            feature_dataset.append(
                (torch.dequantize(logit), target))

        logging.info(
            "[Server #%d] Dequantized features.",
            self.server_id)

        return feature_dataset

