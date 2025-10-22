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
    lora,
    mlx_fedavg,
    split_learning,
)
from plato.config import Config

registered_algorithms = {
    "fedavg": fedavg.Algorithm,
    "fedavg_gan": fedavg_gan.Algorithm,
    "fedavg_personalized": fedavg_personalized.Algorithm,
    "fedavg_lora": lora.Algorithm,
    "mlx_fedavg": mlx_fedavg.Algorithm,
    "split_learning": split_learning.Algorithm,
}


def _resolve_algorithm_type(algorithm_config) -> str:
    """Resolve algorithm type supporting framework shortcuts."""
    algo_type = getattr(algorithm_config, "type", None)
    framework = getattr(algorithm_config, "framework", "")

    if not algo_type and framework:
        if framework.lower() == "mlx":
            return "mlx_fedavg"
    return algo_type


def get(trainer=None):
    """Get the algorithm with the provided type."""
    algorithm_config = Config().algorithm
    algorithm_type = _resolve_algorithm_type(algorithm_config)

    if algorithm_type in registered_algorithms:
        logging.info("Algorithm: %s", algorithm_type)
        registered_alg = registered_algorithms[algorithm_type](trainer)
        return registered_alg
    else:
        raise ValueError(f"No such algorithm: {algorithm_type}")
