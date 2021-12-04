"""
Implements a Processor for applying local differential privacy using randomized response.
"""
import logging
from typing import Any

import torch
from plato.config import Config
from plato.processors import base
from plato.utils import unary_encoding


class Processor(base.Processor):
    """
    Implements a Processor for applying local differential privacy using randomized response.
    """
    def __init__(self, client_id=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for applying randomized response as the
        local differential privacy mechanism.
        """
        if Config().algorithm.epsilon is None:
            return data

        _randomize = getattr(self.trainer, "randomize", None)
        epsilon = Config().algorithm.epsilon
        output = []

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

            output.append((logits, targets))

        logging.info(
            "[Client #%d] Local differential privacy (using randomized response) applied.",
            self.client_id)

        return output
