"""
FedAvg aggregation strategy.

Implements the standard weighted averaging used by most federated learning
algorithms.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Dict, List, Any

import numpy as np

from plato.servers.strategies.base import AggregationStrategy, ServerContext

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None


class FedAvgAggregationStrategy(AggregationStrategy):
    """
    Standard Federated Averaging aggregation.

    Performs weighted averaging of client deltas based on the number of samples
    each client trained on.
    """

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """Aggregate using weighted average by sample count."""
        total_samples = sum(update.report.num_samples for update in updates)

        avg_update: Any = None
        for i, delta in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples
            weight = num_samples / total_samples if total_samples > 0 else 0.0

            avg_update = self._accumulate_weighted(avg_update, delta, weight, context)

            await asyncio.sleep(0)

        return avg_update

    def _accumulate_weighted(
        self,
        target: Any,
        value: Any,
        weight: float,
        context: ServerContext,
    ) -> Any:
        """Accumulate weighted values into the target structure and return it."""
        if value is None or weight == 0.0:
            return target

        if isinstance(value, dict):
            base = target if isinstance(target, dict) and target is not None else {}
            for key, item in value.items():
                base[key] = self._accumulate_weighted(
                    base.get(key), item, weight, context
                )
            return base

        if isinstance(value, np.ndarray):
            base = target if isinstance(target, np.ndarray) else np.zeros_like(value)
            base += value * weight
            return base

        if torch is not None and isinstance(value, torch.Tensor):
            base = target if isinstance(target, torch.Tensor) else torch.zeros_like(value)
            base += value * weight
            return base

        if hasattr(value, "shape"):
            base = target
            if base is None:
                try:
                    base = context.trainer.zeros(value.shape)
                except (AttributeError, TypeError, ValueError):
                    base = np.zeros(value.shape, dtype=getattr(value, "dtype", None))
            if hasattr(base, "__iadd__"):
                base += value * weight
                return base
            return base + value * weight

        base = 0.0 if target is None else target
        return base + value * weight
