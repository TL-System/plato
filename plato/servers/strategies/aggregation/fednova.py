"""
FedNova aggregation strategy.

Implements the FedNova normalization to handle heterogeneous local epochs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional

from plato.config import Config
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedNovaAggregationStrategy(AggregationStrategy):
    """Aggregate deltas with FedNova normalization."""

    @staticmethod
    def _resolve_local_epochs(
        update: SimpleNamespace,
        default_epochs: Optional[float],
    ) -> float:
        """Determine how many local epochs a client ran for its update."""
        report = update.report

        epochs = getattr(report, "epochs", None)
        if epochs is None:
            epochs = getattr(report, "num_epochs", None)

        if epochs is None:
            metrics = getattr(report, "metrics", None)
            if isinstance(metrics, dict):
                epochs = metrics.get("epochs") or metrics.get("num_epochs")

        if epochs is None:
            epochs = default_epochs

        if epochs is None:
            epochs = 1

        try:
            epochs_value = float(epochs)
        except (TypeError, ValueError):
            epochs_value = 1.0

        return max(epochs_value, 1.0)

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        total_samples = sum(update.report.num_samples for update in updates)

        default_epochs = None
        trainer = getattr(context, "trainer", None)
        if trainer is not None and hasattr(trainer, "epochs"):
            default_epochs = trainer.epochs
        else:
            try:
                default_epochs = getattr(Config().trainer, "epochs", None)
            except SystemExit:
                default_epochs = None
            except Exception:  # Config might not be fully initialised
                default_epochs = None

        local_epochs = [
            self._resolve_local_epochs(update, default_epochs) for update in updates
        ]

        avg_update = {
            name: context.trainer.zeros(delta.shape)
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
