"""
FedBuff aggregation strategy.

Applies uniform averaging to buffered asynchronous updates.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Dict, List

from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedBuffAggregationStrategy(AggregationStrategy):
    """Aggregate buffered deltas using equal weights."""

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        if not deltas_received:
            return {}

        total_updates = len(deltas_received)
        weight = 1.0 / total_updates if total_updates > 0 else 0.0

        avg_update = {
            name: context.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for delta in deltas_received:
            for name, value in delta.items():
                avg_update[name] += value * weight

            await asyncio.sleep(0)

        return avg_update
