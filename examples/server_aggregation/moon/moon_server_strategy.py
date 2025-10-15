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
from typing import Deque, Dict, List

import torch

from plato.servers.strategies.base import AggregationStrategy, ServerContext


class MoonAggregationStrategy(AggregationStrategy):
    """Weighted aggregation with a bounded history of global weight snapshots."""

    def __init__(self, history_size: int = 5):
        super().__init__()
        self.history_size = history_size
        self.global_history: Deque[Dict[str, torch.Tensor]] = deque(maxlen=history_size)

    def setup(self, context: ServerContext) -> None:
        """Initialise server-side memory."""
        context.state.setdefault("moon_global_history", self.global_history)

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict[str, torch.Tensor]],
        context: ServerContext,
    ) -> Dict[str, torch.Tensor]:
        """Apply sample-weighted averaging of client updates."""
        # Cache current global weights before updating.
        if context.algorithm is not None:
            baseline = context.algorithm.extract_weights()
            snapshot = {
                name: weights.clone().cpu() for name, weights in baseline.items()
            }
            self.global_history.append(snapshot)

        total_samples = sum(update.report.num_samples for update in updates)
        if total_samples == 0:
            total_samples = 1

        aggregated = {
            name: context.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for update, delta in zip(updates, deltas_received):
            weight = update.report.num_samples / total_samples
            for name, value in delta.items():
                aggregated[name] += value * weight

        return aggregated
