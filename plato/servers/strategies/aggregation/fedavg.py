"""
FedAvg aggregation strategy.

Implements the standard weighted averaging used by most federated learning
algorithms.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Dict, List

from plato.servers.strategies.base import AggregationStrategy, ServerContext


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

        avg_update = {
            name: context.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, delta in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples
            weight = num_samples / total_samples if total_samples > 0 else 0.0

            for name, value in delta.items():
                avg_update[name] += value * weight

            await asyncio.sleep(0)

        return avg_update
