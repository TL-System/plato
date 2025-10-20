"""
FedAtt-specific algorithm helpers encapsulating PyTorch operations.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """Algorithm that provides attentive aggregation utilities for FedAtt."""

    def attentive_aggregate_weights(
        self,
        baseline_weights: Mapping[str, torch.Tensor],
        weights_received: Sequence[Mapping[str, torch.Tensor]],
        *,
        epsilon: float,
        magnitude: float,
    ) -> OrderedDict[str, torch.Tensor]:
        """Aggregate client weights using the FedAtt attention mechanism."""
        if not weights_received:
            return OrderedDict(
                (name, tensor.clone()) for name, tensor in baseline_weights.items()
            )

        deltas = []
        for weight in weights_received:
            delta = OrderedDict()
            for name, baseline in baseline_weights.items():
                delta[name] = weight[name] - baseline
            deltas.append(delta)

        att_update: OrderedDict[str, torch.Tensor] = OrderedDict()

        for name, baseline in baseline_weights.items():
            baseline_device = baseline.device
            delta_norms = torch.stack(
                [update[name].detach().float().norm() for update in deltas]
            )
            attention = F.softmax(delta_norms, dim=0)

            aggregated = torch.zeros_like(
                baseline, dtype=torch.float32, device=baseline_device
            )
            for weight_idx, update in enumerate(deltas):
                delta_tensor = (
                    update[name]
                    .detach()
                    .to(device=baseline_device, dtype=torch.float32)
                )
                aggregated += -delta_tensor * attention[weight_idx]

            noise = torch.randn_like(aggregated) * magnitude
            update_tensor = -aggregated * epsilon + noise
            att_update[name] = update_tensor.to(dtype=baseline.dtype)

        updated_weights = OrderedDict()
        for name, baseline in baseline_weights.items():
            updated_weights[name] = baseline + att_update[name]

        return updated_weights
