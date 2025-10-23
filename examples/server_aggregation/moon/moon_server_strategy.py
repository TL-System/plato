"""
MOON aggregation strategy leveraging the new server strategy API.

The aggregation follows weighted averaging but additionally retains a rolling
history of global models so the server can provide broader context or metrics
if required by downstream tooling.

Reference:
Qinbin Li, Bingsheng He, and Dawn Song.
"Model-Contrastive Federated Learning." CVPR 2021.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Deque

from plato.servers.strategies.base import AggregationStrategy, ServerContext


class MoonAggregationStrategy(AggregationStrategy):
    """Weighted aggregation with a bounded history of global weight snapshots."""

    def __init__(self, history_size: int = 5):
        super().__init__()
        self.history_size = history_size
        self.global_history: Deque[dict] = deque(maxlen=history_size)

    def setup(self, context: ServerContext) -> None:
        """Initialise server-side memory."""
        context.state.setdefault("moon_global_history", self.global_history)

    async def aggregate_deltas(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[dict],
        context: ServerContext,
    ) -> dict:
        """Apply sample-weighted averaging of client updates."""
        # Cache current global weights before updating (delegated to algorithm).
        algorithm = getattr(context, "algorithm", None)
        if algorithm is None:
            raise RuntimeError("MOON requires an algorithm instance in context.")
        baseline = algorithm.extract_weights()
        snapshot = algorithm.moon_snapshot(baseline)
        self.global_history.append(snapshot)

        total_samples = sum(update.report.num_samples for update in updates)
        if total_samples == 0:
            total_samples = 1

        # Perform weighted aggregation via algorithm implementation.
        if not hasattr(algorithm, "moon_aggregate_deltas"):
            raise RuntimeError("Algorithm does not provide MOON aggregation method.")
        return algorithm.moon_aggregate_deltas(updates, deltas_received)
