"""
Pisces aggregation strategy.

Applies staleness-aware weighting to client updates.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from plato.config import Config
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class PiscesAggregationStrategy(AggregationStrategy):
    """Aggregate deltas with polynomial staleness decay."""

    def __init__(self, staleness_factor: float = 1.0, history_window: int = 5):
        super().__init__()
        self.staleness_factor = staleness_factor
        self.history_window = history_window
        self.client_staleness: Dict[int, List[float]] = {}

    def setup(self, context: ServerContext) -> None:
        try:
            if hasattr(Config().server, "staleness_factor"):
                self.staleness_factor = Config().server.staleness_factor
            if hasattr(Config().server, "history_window"):
                self.history_window = Config().server.history_window
        except ValueError:
            pass

        total_clients = context.total_clients
        if total_clients:
            self.client_staleness = {
                client_id: [] for client_id in range(1, total_clients + 1)
            }

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        if not updates or not deltas_received:
            return {}

        total_samples = sum(update.report.num_samples for update in updates)

        avg_update = {
            name: context.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, delta in enumerate(deltas_received):
            client_id = updates[i].client_id
            report = updates[i].report
            num_samples = report.num_samples
            staleness = getattr(updates[i], "staleness", 0.0)

            self.client_staleness.setdefault(client_id, []).append(staleness)
            staleness_factor = self._calculate_staleness_factor(client_id)
            weight = (
                (num_samples / total_samples) * staleness_factor
                if total_samples > 0
                else 0.0
            )

            for name, value in delta.items():
                avg_update[name] += value * weight

            await asyncio.sleep(0)

        return avg_update

    def _calculate_staleness_factor(self, client_id: int) -> float:
        history = self.client_staleness.get(client_id, [])
        if not history:
            return 1.0

        recent_history = history[-self.history_window :]
        staleness = float(np.mean(recent_history))
        return 1.0 / pow(staleness + 1.0, self.staleness_factor)
