"""Aggregation strategy for training GANs in federated settings."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedAvgGanAggregationStrategy(AggregationStrategy):
    """Weighted averaging for GAN generator and discriminator updates."""

    async def aggregate_deltas(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[tuple[dict, dict]],
        context: ServerContext,
    ) -> tuple[dict, dict]:
        """Aggregate generator and discriminator deltas with sample weighting."""

        total_samples = sum(update.report.num_samples for update in updates)
        server_obj = getattr(context, "server", None)
        if server_obj is None:
            raise AttributeError("GAN aggregation requires a server context.")
        server = cast(Any, server_obj)
        server.total_samples = total_samples

        trainer_obj = getattr(context, "trainer", None)
        zeros_fn: Callable[[Any], Any] | None = (
            cast(Callable[[Any], Any], trainer_obj.zeros)
            if trainer_obj is not None and hasattr(trainer_obj, "zeros")
            else None
        )

        gen_avg_update = {
            name: zeros_fn(weights.shape)
            if zeros_fn is not None
            else np.zeros_like(weights)
            for name, weights in deltas_received[0][0].items()
        }
        disc_avg_update = {
            name: zeros_fn(weights.shape)
            if zeros_fn is not None
            else np.zeros_like(weights)
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
