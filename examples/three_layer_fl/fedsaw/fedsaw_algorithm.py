"""
FedSaw-specific algorithm helpers that encapsulate framework-dependent logic.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Mapping, MutableMapping

import torch
from torch import nn
from torch.nn.utils import prune

from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """Algorithm utilities for FedSaw that operate on PyTorch tensors."""

    @staticmethod
    def compute_weight_updates(
        previous_weights: Mapping[str, torch.Tensor],
        new_weights: Mapping[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        """Compute the weight deltas between two model snapshots."""
        deltas: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, new_weight in new_weights.items():
            deltas[name] = new_weight - previous_weights[name]
        return deltas

    def prune_weight_updates(
        self,
        updates: Mapping[str, torch.Tensor],
        *,
        amount: float,
        method: str = "l1",
    ) -> OrderedDict[str, torch.Tensor]:
        """Apply global unstructured pruning to a set of weight updates."""
        if amount <= 0 or not updates:
            return OrderedDict(
                (name, tensor.detach().cpu()) for name, tensor in updates.items()
            )

        pruning_method = (
            prune.RandomUnstructured if method == "random" else prune.L1Unstructured
        )

        # Clone the reference model to host update tensors for pruning.
        delta_model = copy.deepcopy(self.model.cpu())
        cpu_updates: MutableMapping[str, torch.Tensor] = OrderedDict(
            (name, tensor.detach().cpu()) for name, tensor in updates.items()
        )
        delta_model.load_state_dict(cpu_updates, strict=True)

        parameters_to_prune = [
            (module, "weight")
            for _, module in delta_model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        if not parameters_to_prune:
            return OrderedDict(delta_model.state_dict())

        prune.global_unstructured(
            parameters_to_prune, pruning_method=pruning_method, amount=amount
        )
        for module, name in parameters_to_prune:
            prune.remove(module, name)

        return OrderedDict(
            (name, tensor.detach().cpu())
            for name, tensor in delta_model.state_dict().items()
        )

    @staticmethod
    def compute_weight_difference(
        weight_updates: Mapping[str, torch.Tensor],
        *,
        num_samples: int,
        total_samples: int,
    ) -> float:
        """Calculate the scaled norm of weight updates for pruning heuristics."""
        if not weight_updates:
            return 0.0

        scaled_norm = 0.0
        for tensor in weight_updates.values():
            scaled_norm += tensor.detach().float().norm().item()

        if total_samples <= 0:
            return scaled_norm

        return scaled_norm * (num_samples / total_samples)
