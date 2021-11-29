"""
Processor for randomized response.
Only used for features in mistnet in pytorch.
"""
from typing import Any
import logging

import torch

from plato.processors import base
from plato.utils import unary_encoding


class Processor(base.Processor):
    """
    Processor class.
    Base Processor class implementation do nothing on the data.
    """
    def __init__(self,
                 *args,
                 trainer=None,
                 epsilon=None,
                 client_id=None,
                 **kwargs) -> None:
        """Constructor for Processor"""
        self.trainer = trainer
        self.epsilon = epsilon
        self.client_id = client_id

    def process(self, data: Any) -> Any:
        """
        Data processing implementation.
        Implement this method while inheriting the class.
        """
        if self.epsilon is None:
            return data

        _randomize = getattr(self.trainer, "randomize", None)
        epsilon = self.epsilon
        new_data = []

        for logits, targets in data:
            logits = logits.detach().numpy()
            logits = unary_encoding.encode(logits)
            if callable(_randomize):
                logits = self.trainer.randomize(logits, targets, epsilon)
            else:
                logits = unary_encoding.randomize(logits, epsilon)
            if self.trainer.device != 'cpu':
                logits = torch.from_numpy(logits.astype('float16'))
            else:
                logits = torch.from_numpy(logits.astype('float32'))

            new_data.append((logits, targets))

        logging.info("[Client #%d] Randomized response applied.",
                     self.client_id)

        return data
