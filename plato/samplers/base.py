"""
Base class for sampling data so that a dataset can be divided across the clients.
"""

import os
from abc import abstractmethod

import numpy as np

from plato.config import Config


class Sampler:
    """Base class for data samplers so that the dataset is divided into
    partitions across the clients."""

    def __init__(self, client_id: int | str | None = None):
        if client_id is not None:
            total_clients = Config().clients.total_clients
            if not 1 <= int(client_id) <= total_clients:
                raise ValueError(f"client_id must be between 1 and {total_clients}.")
        if hasattr(Config().data, "random_seed"):
            # Keeping random seed the same across the clients
            # so that the experiments are reproducible
            self.random_seed = Config().data.random_seed
        else:
            # The random seed will be different across different
            # runs if it is not provided.
            self.random_seed = os.getpid()

        # RandomState preserves the historical seeded sequence without changing
        # NumPy's process-global RNG used by model initialization and training.
        self.rng = np.random.RandomState(self.random_seed)

    @abstractmethod
    def get(self):
        """Obtains an instance of the sampler."""

    @abstractmethod
    def num_samples(self):
        """Returns the length of the dataset after sampling."""
