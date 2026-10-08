"""Aggregation strategy for homomorphic-encrypted FedAvg."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, cast

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
    ) -> dict | None:
        if len(updates) != len(weights_received):
            raise ValueError("HE report/payload cardinality mismatch.")
        if not weights_received:
            return None

        # MaskCrypt alternates between mask tensors and encrypted weights.
        # Skip aggregation when the payload lacks the serialized model structure.
        mappings = [isinstance(payload, Mapping) for payload in weights_received]
        if not any(mappings):
            return None
        if not all(mappings):
            raise ValueError("Cannot mix HE model payloads and mask payloads.")
        counts = [update.report.num_samples for update in updates]
        total_samples = sum(counts)
        if any(
            not math.isfinite(count) or count < 0 for count in counts
        ) or not math.isfinite(total_samples):
            raise ValueError("HE sample counts must be finite and nonnegative.")

        server_obj = getattr(context, "server", None)
        if server_obj is None:
            raise AttributeError("HE aggregation requires a server context.")
        server = cast(Any, server_obj)

        fedavg_hybrid = getattr(server, "_fedavg_hybrid", None)
        if fedavg_hybrid is None:
            raise AttributeError("Server must implement '_fedavg_hybrid'.")

        weight_shapes = getattr(server, "weight_shapes", None)
        para_nums = getattr(server, "para_nums", None)
        if weight_shapes is None or para_nums is None:
            raise AttributeError(
                "Server must expose 'weight_shapes' and 'para_nums' for HE aggregation."
            )

        for payload in weights_received:
            homo_enc.validate_encrypted_model(payload, sum(para_nums.values()))
        if total_samples == 0:
            return baseline_weights

        aggregated = fedavg_hybrid(updates, weights_received)
        decrypted_weights = homo_enc.decrypt_weights(
            aggregated, weight_shapes, para_nums
        )

        encrypted_part = aggregated.get("encrypted_weights")
        if encrypted_part is not None and not isinstance(encrypted_part, bytes):
            aggregated["encrypted_weights"] = encrypted_part.serialize()
        server.encrypted_model = aggregated

        return decrypted_weights
