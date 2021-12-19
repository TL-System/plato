"""
Implements a generalized Processor for applying operations onto MistNet PyTorch features.
"""
from typing import Any, Callable

import torch
from plato.processors import base


class Processor(base.Processor):
    """
    Implements a generalized Processor for applying operations onto MistNet PyTorch features.
    """
    def __init__(self,
                 method: Callable = lambda x, y: (x, y),
                 client_id=None,
                 use_numpy=True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id
        self.method = method
        self.use_numpy = use_numpy

    def process(self, data: Any) -> Any:
        """
        Implements a generalized Processor for applying operations onto MistNet PyTorch features.
        """

        output = []

        for logits, targets in data:
            if self.use_numpy:
                logits = logits.detach().numpy()

            logits, targets = self.method(logits, targets)

            if self.use_numpy:
                if self.trainer.device != 'cpu':
                    logits = torch.from_numpy(logits.astype('float16'))
                else:
                    logits = torch.from_numpy(logits.astype('float32'))

            output.append((logits, targets))

        return output
