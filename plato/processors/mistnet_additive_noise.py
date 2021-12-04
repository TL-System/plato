"""
Implements a Processor for applying local differential privacy using additive noise mechanism.
"""
import logging
from typing import Any
import numpy

import torch
from plato.processors import base


class Processor(base.Processor):
    """
    Implements a Processor for applying local differential privacy using additive noise mechanism.
    """

    methods = {
        "gaussian":
        numpy.random.normal,
        "laplace":
        numpy.random.laplace,
        "exponantial":
        lambda logits, scale: logits + numpy.random.exponential(
            scale, logits.shape)
    }

    def __init__(self,
                 *args,
                 method="",
                 scale=None,
                 client_id=None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.scale = scale
        self.client_id = client_id
        self.method = method
        self._method = Processor.methods[method]

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for applying random response as the local differential privacy
        mechanism.
        """

        scale = self.scale

        new_data = []

        for logits, targets in data:
            logits = logits.detach().numpy()
            logits = self._method(logits, scale)
            if self.trainer.device != 'cpu':
                logits = torch.from_numpy(logits.astype('float16'))
            else:
                logits = torch.from_numpy(logits.astype('float32'))

            new_data.append((logits, targets))

        logging.info(
            "[Client #%d] Local differential privacy (using %s mechanism) applied.",
            self.method_str, self.client_id)

        return data
