"""
Implements a Processor for applying local differential privacy using laplace mechanism.
"""

from plato.processors import mistnet_additive_noise


class Processor(mistnet_additive_noise.Processor):
    """
    Implements a Processor for applying local differential privacy using laplace mechanism.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, method="laplace", **kwargs)
