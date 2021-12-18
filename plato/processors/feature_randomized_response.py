"""
Implements a Processor for applying local differential privacy using randomized response.
"""
import logging
from typing import Any

from plato.config import Config
from plato.processors import feature
from plato.utils import unary_encoding


class Processor(feature.Processor):
    """
    Implements a Processor for applying local differential privacy using randomized response.
    """
    def __init__(self, **kwargs) -> None:
        def func(logits, targets):
            if Config().algorithm.epsilon is None:
                return logits, targets

            _randomize = getattr(self.trainer, "randomize", None)
            epsilon = Config().algorithm.epsilon

            logits = unary_encoding.encode(logits)
            if callable(_randomize):
                logits = self.trainer.randomize(logits, targets, epsilon)
            else:
                logits = unary_encoding.randomize(logits, epsilon)

            return logits, targets

        super().__init__(method=func, **kwargs)

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for applying randomized response as the
        local differential privacy mechanism.
        """

        output = super().process(data)

        logging.info(
            "[Client #%d] Local differential privacy (using randomized response) applied.",
            self.client_id)

        return output
