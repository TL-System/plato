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

    def __init__(
        self, client_id: int | str | None = None, *, edge_evaluation: bool = False
    ):
        if client_id is not None:
            config = Config()
            total_clients = config.clients.total_clients
            numeric_id = int(client_id)
            # Full-pool evaluation samplers can retain an edge's physical ID
            # (and seeded bias). Indexed client partitions do not opt in.
            assigned_edge = (
                edge_evaluation
                and config.is_edge_server()
                and config.args.id == numeric_id
                and total_clients < numeric_id
                <= total_clients + getattr(config.algorithm, "total_silos", 0)
            )
            if not 1 <= numeric_id <= total_clients and not assigned_edge:
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
