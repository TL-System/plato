"""
Implements a Processor for applying quantization to MistNet PyTorch features.
"""
import logging
from typing import Any
import torch

from plato.processors import feature


class Processor(feature.Processor):
    """
    Implements a Processor for applying quantization to MistNet PyTorch features.
    """
    def __init__(self,
                 scale=0.1,
                 zero_point=10,
                 dtype=torch.quint8,
                 **kwargs) -> None:
        def func(logits, targets):
            logits = torch.quantize_per_tensor(logits, scale, zero_point,
                                               dtype)
            return logits, targets

        super().__init__(method=func, use_numpy=False, **kwargs)

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for applying quantization to MistNet PyTorch features.
        """

        output = super().process(data)

        logging.info("[Client #%d] Quantization applied.", self.client_id)

        return output
