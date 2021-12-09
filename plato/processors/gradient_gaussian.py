"""
A Processor for applying local differential privacy using the Gaussian mechanism.
"""
import math
import torch

from plato.processors import gradient_additive_noise
from plato.config import Config


class Processor(gradient_additive_noise.Processor):
    """
    Implements a Processor for applying local differential privacy using the Gaussian mechanism.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.delta = Config().algorithm.dp_delta
        self.epsilon = Config().algorithm.dp_epsilon

    def compute_additive_noise(self, gradient, clipping_bound):
        """ Computes Gaussian noise. """

        scale = math.sqrt(
            2 * math.log(1.25 / self.delta)) * clipping_bound / self.epsilon

        additive_noise = torch.normal(mean=0.0, std=scale, size=gradient.shape)

        return additive_noise
