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

    def __init__(self, method="", scale=None, **kwargs) -> None:

        self._method = method
        func = lambda logits, targets: (Processor.methods[method]
                                        (logits, scale), targets)
        super().__init__(method=func, **kwargs)

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for applying randomized response as the local differential privacy
        mechanism.
        """

        output = super().process(data)

        logging.info(
            "[Client #%d] Local differential privacy (using %s mechanism) applied.",
            self.client_id, self._method)

        return output
