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

    @staticmethod
    def _cast_tensor_like(
        tensor: torch.Tensor,
        reference: torch.Tensor,
        tensor_name: str = "tensor",
    ) -> torch.Tensor:
        """Cast a tensor to match a reference dtype (handles bool/int safely)."""
        if tensor.dtype == reference.dtype:
            return tensor

        if torch.is_floating_point(reference):
            return tensor.to(reference.dtype)

        if reference.dtype == torch.bool:
            if torch.is_floating_point(tensor):
                return tensor >= 0.5
            return tensor.ne(0)

        if torch.is_floating_point(tensor):
            return torch.round(tensor).to(reference.dtype)

        return tensor.to(reference.dtype)

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

        reference = deltas_received[0]
        aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, delta in reference.items():
            if torch.is_floating_point(delta):
                aggregated[name] = torch.zeros_like(delta)
            else:
                aggregated[name] = torch.zeros_like(
                    delta, dtype=torch.get_default_dtype()
                )

        for u, delta in zip(updates, deltas_received):
            w = (u.report.num_samples or 0) / total
            if w == 0.0:
                continue
            for name, value in delta.items():
                target = aggregated[name]
                if not torch.is_floating_point(value) or value.dtype != target.dtype:
                    value = value.to(target.dtype)
                aggregated[name] = target + value * w

        for name, ref_tensor in reference.items():
            aggregated[name] = self._cast_tensor_like(aggregated[name], ref_tensor)

        return aggregated
