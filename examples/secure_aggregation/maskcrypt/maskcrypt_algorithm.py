"""
MaskCrypt-specific algorithm helpers.
"""

from __future__ import annotations

import os
import pickle
from typing import Iterable, List, Mapping, Optional, Sequence

import torch

from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """Algorithm that provides utilities required by the MaskCrypt workflow."""

    @staticmethod
    def build_consensus_mask(
        proposals: Sequence[Sequence[int]],
    ) -> List[int]:
        """Interleave client proposals and return a de-duplicated mask."""
        if not proposals:
            return []

        interleaved: List[int] = []
        max_length = max((len(proposal) for proposal in proposals), default=0)

        for index in range(max_length):
            for proposal in proposals:
                if index < len(proposal):
                    interleaved.append(int(proposal[index]))

        seen = set()
        mask: List[int] = []
        for value in interleaved:
            if value not in seen:
                seen.add(value)
                mask.append(value)

        return mask

    @staticmethod
    def flatten_weights(weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Flatten a model state dict into a single CPU tensor."""
        if not weights:
            return torch.tensor([])

        flats = [tensor.detach().cpu().flatten() for tensor in weights.values()]
        return torch.cat(flats)

    @staticmethod
    def flatten_gradients(gradients: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Flatten model gradients into a single CPU tensor."""
        if not gradients:
            return torch.tensor([])

        flats = [tensor.detach().cpu().flatten() for tensor in gradients.values()]
        return torch.cat(flats)

    @staticmethod
    def prepare_exposed_weights(
        estimate: Optional[Sequence[float]],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Convert stored estimates into a tensor aligned with the latest weights."""
        if estimate is None or len(estimate) == 0:
            return torch.zeros_like(reference, dtype=torch.float32)

        exposed = torch.tensor(estimate, dtype=torch.float32)
        if exposed.numel() != reference.numel():
            # Resize conservatively to match the reference vector.
            if exposed.numel() < reference.numel():
                padding = torch.zeros(reference.numel() - exposed.numel())
                exposed = torch.cat([exposed, padding])
            else:
                exposed = exposed[: reference.numel()]
        return exposed

    @staticmethod
    def store_plain_weights(path: str, flattened_weights: torch.Tensor) -> None:
        """Persist the latest flattened weights for external analysis."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as plain_file:
            pickle.dump(flattened_weights.detach().cpu(), plain_file)

    @staticmethod
    def compute_mask(
        *,
        latest_flat: torch.Tensor,
        gradients_flat: torch.Tensor,
        exposed_flat: torch.Tensor,
        encrypt_ratio: float,
        random_mask: bool,
    ) -> torch.Tensor:
        """Compute the selective encryption mask for a client."""
        num_params = latest_flat.numel()
        mask_len = max(int(encrypt_ratio * num_params), 0)
        if mask_len <= 0 or num_params == 0:
            return torch.tensor([], dtype=torch.long)

        if random_mask:
            indices = torch.randperm(num_params)[:mask_len]
            return indices.to(dtype=torch.long)

        latest = latest_flat.to(dtype=torch.float32)
        gradients = gradients_flat.to(dtype=torch.float32)
        exposed = exposed_flat.to(dtype=torch.float32)

        delta = exposed - latest
        product = delta * gradients

        _, indices = torch.sort(product, descending=True)
        return indices[:mask_len].clone().detach().to(dtype=torch.long)
