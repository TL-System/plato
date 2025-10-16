"""
The registry for algorithms that contains framework-specific implementations.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

import logging

from plato.algorithms import (
    fedavg,
    fedavg_gan,
    fedavg_personalized,
    split_learning,
)
from plato.config import Config

registered_algorithms = {
    "fedavg": fedavg.Algorithm,
    "fedavg_gan": fedavg_gan.Algorithm,
    "fedavg_personalized": fedavg_personalized.Algorithm,
    "split_learning": split_learning.Algorithm,
}


def get(trainer=None):
    """Get the algorithm with the provided type."""
    algorithm_type = Config().algorithm.type

    if algorithm_type in registered_algorithms:
        logging.info("Algorithm: %s", algorithm_type)
        registered_alg = registered_algorithms[algorithm_type](trainer)
        return registered_alg
    else:
        raise ValueError(f"No such algorithm: {algorithm_type}")
