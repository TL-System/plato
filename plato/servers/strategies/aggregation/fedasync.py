"""
FedAsync aggregation strategy.

Supports staleness-aware mixing for asynchronous federated learning.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast

from plato.config import Config
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedAsyncAggregationStrategy(AggregationStrategy):
    """Aggregate updates with configurable staleness-aware mixing."""

    def __init__(
        self,
        mixing_hyperparameter: float = 0.9,
        adaptive_mixing: bool = False,
        staleness_func_type: str = "constant",
        staleness_func_params: dict | None = None,
    ):
        super().__init__()
        self.mixing_hyperparam = mixing_hyperparameter
        self.adaptive_mixing = adaptive_mixing
        self.staleness_func_type = staleness_func_type.lower()
        self.staleness_func_params = staleness_func_params or {}

    def setup(self, context: ServerContext) -> None:
        server_config = getattr(Config(), "server", None)

        if server_config is None:
            logging.warning(
                "FedAsync: No server configuration found; using default hyperparameters."
            )
            return

        mixing_value = getattr(server_config, "mixing_hyperparameter", None)
        if mixing_value is None:
            logging.warning(
                "FedAsync: Variable mixing hyperparameter is required for the FedAsync server."
            )
        else:
            try:
                mixing_value = float(mixing_value)
            except (TypeError, ValueError):
                logging.warning(
                    "FedAsync: Invalid mixing hyperparameter. Unable to cast %s to float.",
                    mixing_value,
                )
            else:
                if 0 < mixing_value < 1:
                    self.mixing_hyperparam = mixing_value
                    logging.info(
                        "FedAsync: Mixing hyperparameter is set to %s.",
                        self.mixing_hyperparam,
                    )
                else:
                    logging.warning(
                        "FedAsync: Invalid mixing hyperparameter. "
                        "The hyperparameter needs to be between 0 and 1 (exclusive)."
                    )

        adaptive_value = getattr(server_config, "adaptive_mixing", None)
        if adaptive_value is not None:
            self.adaptive_mixing = bool(adaptive_value)

        staleness_config = getattr(server_config, "staleness_weighting_function", None)
        if staleness_config is not None:
            func_type = getattr(staleness_config, "type", "constant")
            staleness_type = str(func_type).lower()
            params: dict[str, float] = {}

            if staleness_type == "polynomial":
                params["a"] = float(getattr(staleness_config, "a", 1.0))
            elif staleness_type == "hinge":
                params["a"] = float(getattr(staleness_config, "a", 1.0))
                params["b"] = float(getattr(staleness_config, "b", 10))
            elif staleness_type != "constant":
                logging.warning(
                    "FedAsync: Unknown staleness weighting function type '%s'. "
                    "Falling back to constant.",
                    staleness_type,
                )
                staleness_type = "constant"

            self.staleness_func_type = staleness_type
            self.staleness_func_params = params

    async def aggregate_deltas(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[dict],
        context: ServerContext,
    ) -> dict:
        """Fallback delta aggregation using weighted averaging."""
        total_samples = sum(update.report.num_samples for update in updates)

        trainer = getattr(context, "trainer", None)
        if trainer is None or not hasattr(trainer, "zeros"):
            raise AttributeError(
                "FedAsync requires the trainer to provide a 'zeros' method."
            )
        zeros_fn = trainer.zeros

        avg_update = {
            name: zeros_fn(delta.shape) for name, delta in deltas_received[0].items()
        }

        for i, delta in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples
            weight = num_samples / total_samples if total_samples > 0 else 0.0

            for name, value in delta.items():
                avg_update[name] += value * weight

            await asyncio.sleep(0)

        return avg_update

    async def aggregate_weights(
        self,
        updates: list[SimpleNamespace],
        baseline_weights: dict,
        weights_received: list[dict],
        context: ServerContext,
    ) -> dict:
        """Aggregate weights directly with staleness-aware mixing."""
        if not updates:
            return baseline_weights

        client_staleness = getattr(updates[0], "staleness", 0)
        mixing = self.mixing_hyperparam

        if self.adaptive_mixing:
            mixing *= self._staleness_function(client_staleness)

        algorithm = getattr(context, "algorithm", None)
        if algorithm is None or not hasattr(algorithm, "aggregate_weights"):
            raise AttributeError(
                "FedAsync requires an algorithm with 'aggregate_weights'."
            )

        algorithm = cast(Any, algorithm)

        return await algorithm.aggregate_weights(
            baseline_weights, weights_received, mixing=mixing
        )

    def _staleness_function(self, staleness: int) -> float:
        """Calculate staleness weighting factor."""
        if self.staleness_func_type == "constant":
            return 1.0
        if self.staleness_func_type == "polynomial":
            a = self.staleness_func_params.get("a", 1.0)
            return 1 / (staleness + 1) ** a
        if self.staleness_func_type == "hinge":
            a = self.staleness_func_params.get("a", 1.0)
            b = self.staleness_func_params.get("b", 10)
            return 1.0 if staleness <= b else 1 / (a * (staleness - b) + 1)

        logging.warning(
            "FedAsync: Unknown staleness function type '%s'. Using constant.",
            self.staleness_func_type,
        )
        return 1.0
