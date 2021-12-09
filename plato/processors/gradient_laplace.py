"""
A Processor for applying local differential privacy using the Laplace mechanism.
"""
from torch.distributions.laplace import Laplace

from plato.processors import gradient_additive_noise
from plato.config import Config


class Processor(gradient_additive_noise.Processor):
    """
    Implements a Processor for applying local differential privacy using the Laplace mechanism.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.epsilon = Config().algorithm.dp_epsilon

    def compute_additive_noise(self, gradient, clipping_bound):
        """ Computes Laplace noise. """
        scale = clipping_bound / self.epsilon

        additive_noise = Laplace(loc=0, scale=scale).sample(gradient.shape)

        return additive_noise
