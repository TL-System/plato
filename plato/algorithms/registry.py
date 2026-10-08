"""
Registry for algorithms that contains framework-specific implementations.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from plato.algorithms import (
    fedavg,
    fedavg_gan,
    fedavg_personalized,
    lora,
    pfedgraph,
    split_learning,
)
from plato.algorithms.base import Algorithm as AlgorithmBase
from plato.config import Config


def _mlx_fedavg(trainer: Any) -> AlgorithmBase:
    """Resolve the native constructor only when its registry entry is called."""
    from plato.algorithms.mlx_fedavg import Algorithm

    if registered_algorithms.get("mlx_fedavg") is _mlx_fedavg:
        registered_algorithms["mlx_fedavg"] = Algorithm
    return Algorithm(trainer)


registered_algorithms: dict[str, Callable[[Any], AlgorithmBase]] = {
    "fedavg": fedavg.Algorithm,
    "fedavg_gan": fedavg_gan.Algorithm,
    "fedavg_personalized": fedavg_personalized.Algorithm,
    "fedavg_lora": lora.Algorithm,
    "mlx_fedavg": _mlx_fedavg,
    "pfedgraph": pfedgraph.Algorithm,
    "split_learning": split_learning.Algorithm,
}


def _resolve_algorithm_type(algorithm_config: Any) -> str:
    """Resolve algorithm type supporting framework shortcuts."""
    algo_type_obj: Any | None = getattr(algorithm_config, "type", None)
    algo_type = algo_type_obj if isinstance(algo_type_obj, str) else None

    framework_obj: Any | None = getattr(algorithm_config, "framework", "")
    framework = framework_obj if isinstance(framework_obj, str) else ""
    if (framework.lower() == "mlx" and algo_type not in (None, "mlx_fedavg")) or (
        algo_type == "mlx_fedavg" and framework and framework.lower() != "mlx"
    ):
        raise ValueError(
            "MLX algorithm type and framework must select the same backend."
        )

    if not algo_type and framework:
        if framework.lower() == "mlx":
            return "mlx_fedavg"

    if not algo_type:
        raise ValueError("Algorithm type must be specified in the configuration.")
    return algo_type


def get(trainer: Any | None = None) -> AlgorithmBase:
    """Get the algorithm with the provided type."""
    algorithm_config = Config().algorithm
    algorithm_type = _resolve_algorithm_type(algorithm_config)
    if algorithm_type == "mlx_fedavg":
        from plato.trainers.mlx import ComposableMLXTrainer, _ensure_mlx_available

        _ensure_mlx_available()
        if not isinstance(trainer, ComposableMLXTrainer):
            raise TypeError("MLX FedAvg requires a native MLX trainer.")

    if algorithm_type in registered_algorithms:
        logging.info("Algorithm: %s", algorithm_type)
        registered_alg = registered_algorithms[algorithm_type](trainer)
        return registered_alg
    raise ValueError(f"No such algorithm: {algorithm_type}")
