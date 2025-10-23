"""
MOON-specific Algorithm helpers encapsulating PyTorch operations.

Reference:
Qinbin Li, Bingsheng He, and Dawn Song. "Model-Contrastive Federated Learning." CVPR 2021.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Mapping, Sequence

import torch

from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """Algorithm providing MOON aggregation utilities."""

    def moon_snapshot(self, weights: Mapping[str, torch.Tensor]) -> dict:
        """Create a safe snapshot of the provided weights."""
        # Use a deepcopy to avoid in-place mutations on tensors; keep on CPU
        return copy.deepcopy({k: v.detach().cpu() for k, v in weights.items()})

    def moon_aggregate_deltas(
        self,
        updates: Sequence[Any],
        deltas_received: Sequence[Mapping[str, torch.Tensor]],
    ) -> OrderedDict[str, torch.Tensor]:
        """Sample-weighted averaging of client deltas (PyTorch ops)."""
        if not deltas_received:
            return OrderedDict()

        total = sum(u.report.num_samples for u in updates) or 1

        aggregated: OrderedDict[str, torch.Tensor] = OrderedDict(
            (name, torch.zeros_like(delta))
            for name, delta in deltas_received[0].items()
        )

        for u, delta in zip(updates, deltas_received):
            w = (u.report.num_samples or 0) / total
            if w == 0.0:
                continue
            for name, value in delta.items():
                aggregated[name] += value * w

        return aggregated
