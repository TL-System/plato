"""
FedNova aggregation strategy.

Implements the FedNova normalization to handle heterogeneous local epochs.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, cast

import numpy as np

from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedNovaAggregationStrategy(AggregationStrategy):
    """Aggregate deltas with FedNova normalization."""

    async def aggregate_deltas(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[dict],
        context: ServerContext,
    ) -> dict:
        total_samples = sum(update.report.num_samples for update in updates)
        local_epochs = [update.report.epochs for update in updates]

        trainer = getattr(context, "trainer", None)
        zeros_fn: Callable[[Any], Any] | None = (
            cast(Callable[[Any], Any], trainer.zeros)
            if trainer is not None and hasattr(trainer, "zeros")
            else None
        )

        avg_update = {
            name: zeros_fn(delta.shape)
            if zeros_fn is not None
            else np.zeros_like(delta)
            for name, delta in deltas_received[0].items()
        }

        tau_eff = 0.0
        for i, delta in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples
            tau_eff += local_epochs[i] * num_samples / total_samples

        for i, delta in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples
            for name, value in delta.items():
                avg_update[name] += (
                    value
                    * (num_samples / total_samples)
                    * tau_eff
                    / max(local_epochs[i], 1)
                )

        return avg_update
