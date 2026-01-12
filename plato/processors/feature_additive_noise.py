"""
Implements a Processor for applying local differential privacy using additive noise mechanism.
"""

import logging
from typing import Any

import numpy

from plato.processors import feature


class Processor(feature.Processor):
    """
    Implements a Processor for applying local differential privacy using additive noise mechanism.
    """

    methods = {
        "gaussian": numpy.random.normal,
        "laplace": numpy.random.laplace,
    }

    def __init__(self, method: str = "", scale: float | None = None, **kwargs) -> None:
        if method not in Processor.methods:
            raise ValueError(f"Unknown additive noise method: {method}")
        self._method = method
        scale_value = 1.0 if scale is None else float(scale)

        def func(logits, targets):
            return (
                Processor.methods[method](logits, scale_value),
                targets,
            )

        super().__init__(method=func, **kwargs)

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for applying randomized response as the local differential privacy
        mechanism.
        """

        output = super().process(data)

        logging.info(
            "[Client #%d] Local differential privacy (using the %s mechanism) applied.",
            self.client_id,
            self._method,
        )

        return output
