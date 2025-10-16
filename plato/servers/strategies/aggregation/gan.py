"""Aggregation strategy for training GANs in federated settings."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Dict, List, Tuple

from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedAvgGanAggregationStrategy(AggregationStrategy):
    """Weighted averaging for GAN generator and discriminator updates."""

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Tuple[Dict, Dict]],
        context: ServerContext,
    ) -> Tuple[Dict, Dict]:
        """Aggregate generator and discriminator deltas with sample weighting."""

        total_samples = sum(update.report.num_samples for update in updates)
        context.server.total_samples = total_samples

        trainer = context.trainer

        gen_avg_update = {
            name: trainer.zeros(weights.shape)
            for name, weights in deltas_received[0][0].items()
        }
        disc_avg_update = {
            name: trainer.zeros(weights.shape)
            for name, weights in deltas_received[0][1].items()
        }

        for i, (gen_deltas, disc_deltas) in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples
            weight = num_samples / total_samples if total_samples > 0 else 0.0

            for name, delta in gen_deltas.items():
                gen_avg_update[name] += delta * weight

            for name, delta in disc_deltas.items():
                disc_avg_update[name] += delta * weight

            await asyncio.sleep(0)

        return gen_avg_update, disc_avg_update
