"""
Implements a Processor for applying local differential privacy using laplace mechanism.
"""

from plato.processors import feature_additive_noise


class Processor(feature_additive_noise.Processor):
    """
    Implements a Processor for applying local differential privacy using laplace mechanism.
    """
    def __init__(self, epsilon=None, sensitivity=None, **kwargs) -> None:

        scale = sensitivity / epsilon
        super().__init__(method="laplace", scale=scale, **kwargs)
