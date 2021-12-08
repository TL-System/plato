"""
A Processor for applying local differential privacy using Laplace mechanism.
"""
import numpy as np

from plato.processors import gradient_additive_noise
from plato.config import Config


class Processor(gradient_additive_noise.Processor):
    """
    Implement a Processor for applying local differential privacy using Laplace mechanism.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.epsilon = Config().algorithm.dp_epsilon

    def compute_additive_noise(self, gradient, clipping_bound):
        """Compute Laplace noise."""

        scale = clipping_bound / self.epsilon

        additive_noise = np.random.laplace(loc=0.0,
                                           scale=scale,
                                           size=gradient.shape)

        return additive_noise
