"""Aggregation strategy for homomorphic-encrypted FedAvg."""

from __future__ import annotations

from typing import Dict, List

from plato.servers.strategies.base import AggregationStrategy, ServerContext
from plato.utils import homo_enc


class FedAvgHEAggregationStrategy(AggregationStrategy):
    """Aggregate updates that mix encrypted and unencrypted weights."""

    async def aggregate_deltas(
        self,
        updates,
        deltas_received,
        context: ServerContext,
    ):
        raise NotImplementedError(
            "FedAvgHEAggregationStrategy operates on weights directly."
        )

    async def aggregate_weights(
        self,
        updates,
        baseline_weights,
        weights_received,
        context: ServerContext,
    ) -> Dict:
        server = context.server

        aggregated = server._fedavg_hybrid(updates, weights_received)
        server.encrypted_model = aggregated

        decrypted_weights = homo_enc.decrypt_weights(
            aggregated, server.weight_shapes, server.para_nums
        )

        encrypted_part = aggregated.get("encrypted_weights")
        if encrypted_part is not None and not isinstance(encrypted_part, bytes):
            aggregated["encrypted_weights"] = encrypted_part.serialize()

        return decrypted_weights
