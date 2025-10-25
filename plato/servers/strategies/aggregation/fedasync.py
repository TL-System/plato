"""
FedAsync aggregation strategy.

Supports staleness-aware mixing for asynchronous federated learning.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, cast

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
        try:
            if hasattr(Config().server, "mixing_hyperparameter"):
                self.mixing_hyperparam = Config().server.mixing_hyperparameter
            if hasattr(Config().server, "adaptive_mixing"):
                self.adaptive_mixing = Config().server.adaptive_mixing
        except ValueError:
            pass

        logging.info(
            "FedAsync: Mixing hyperparameter set to %s (adaptive=%s)",
            self.mixing_hyperparam,
            self.adaptive_mixing,
        )

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
