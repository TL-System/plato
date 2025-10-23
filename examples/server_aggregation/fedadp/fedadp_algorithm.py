"""
FedAdp-specific Algorithm encapsulating PyTorch/Numpy aggregation logic.

Reference:

H. Wu, P. Wang. "Fast-Convergent Federated Learning with Adaptive Weighting,"
in IEEE Trans. on Cognitive Communications and Networking (TCCN), 2021.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """Algorithm that provides FedAdp adaptive aggregation utilities."""

    def __init__(self, trainer):
        super().__init__(trainer)
        self.alpha: float = 5.0
        self.local_angles: dict[int, float] = {}

    def fedadp_aggregate_deltas(
        self,
        updates: Sequence[Any],
        deltas_received: Sequence[Mapping[str, torch.Tensor]],
        *,
        alpha: float,
        current_round: int,
    ) -> OrderedDict[str, torch.Tensor]:
        """Aggregate client deltas using FedAdp's adaptive weighting.

        Args:
            updates: Client update meta (contains report.num_samples and client_id).
            deltas_received: Per-client weight delta dicts.
            alpha: FedAdp alpha parameter.
            current_round: Current FL round (1-based expected for EWMA).
        """
        self.alpha = alpha

        if not deltas_received:
            return OrderedDict()

        num_samples = [u.report.num_samples for u in updates]
        total_samples = sum(num_samples)
        total_samples = total_samples if total_samples > 0 else 1

        # Global gradients as the sample-weighted average of client deltas
        global_grads: OrderedDict[str, torch.Tensor] = OrderedDict(
            (name, torch.zeros_like(tensor))
            for name, tensor in deltas_received[0].items()
        )
        for idx, delta in enumerate(deltas_received):
            weight = num_samples[idx] / total_samples
            for name, value in delta.items():
                global_grads[name] += value * weight

        # Compute adaptive weighting
        contribs = self._calc_contribution(
            updates, deltas_received, global_grads, current_round
        )

        # Normalize weights with data size and contributions
        weights: list[float] = [0.0] * len(deltas_received)
        denom = 0.0
        for i, c in enumerate(contribs):
            denom += num_samples[i] * math.exp(c)
        if denom == 0.0:
            # Fallback to uniform if degenerate
            weights = [1.0 / len(deltas_received)] * len(deltas_received)
        else:
            for i, c in enumerate(contribs):
                weights[i] = (num_samples[i] * math.exp(c)) / denom

        # Aggregate deltas with the computed weights
        agg: OrderedDict[str, torch.Tensor] = OrderedDict(
            (name, torch.zeros_like(tensor))
            for name, tensor in deltas_received[0].items()
        )
        for idx, delta in enumerate(deltas_received):
            w = weights[idx]
            if w == 0.0:
                continue
            for name, value in delta.items():
                agg[name] += value * w

        return agg

    # ---- helpers ----
    def _calc_contribution(
        self,
        updates: Sequence[Any],
        deltas_received: Sequence[Mapping[str, torch.Tensor]],
        global_grads: Mapping[str, torch.Tensor],
        current_round: int,
    ) -> list[float]:
        """Calculate node contribution from angle between local and global grads."""
        num_clients = len(deltas_received)
        angles = [0.0] * num_clients

        global_vec = self._flatten_grads(global_grads)
        g_norm = np.linalg.norm(global_vec)

        for idx, delta in enumerate(deltas_received):
            local_vec = self._flatten_grads(delta)
            l_norm = np.linalg.norm(local_vec)
            if g_norm == 0.0 or l_norm == 0.0:
                angles[idx] = 0.0
            else:
                inner = float(np.inner(global_vec, local_vec))
                cos = inner / (g_norm * l_norm)
                angles[idx] = float(np.arccos(np.clip(cos, -1.0, 1.0)))

        t = current_round if current_round > 0 else 1
        contribs = [0.0] * num_clients
        alpha = self.alpha

        for idx, angle in enumerate(angles):
            client_id = updates[idx].client_id
            prev = self.local_angles.get(client_id, angle)
            ewma = ((t - 1) / t) * prev + (1 / t) * angle
            self.local_angles[client_id] = ewma

            contribs[idx] = alpha * (1 - math.exp(-math.exp(-alpha * (ewma - 1))))

        return contribs

    @staticmethod
    def _flatten_grads(grads: Mapping[str, torch.Tensor]) -> np.ndarray:
        """Flatten a dict of tensors into a 1-D numpy vector.

        Preserves the original behaviour where the first tensor is appended as-is
        and subsequent tensors are scaled by -1/lr before concatenation.
        """
        # Resolve LR from config (fallback to 1.0 to avoid div-by-zero)
        lr = 1.0
        try:
            lr = float(Config().parameters.optimizer.lr)
        except Exception:
            lr = 1.0

        items = list(sorted(grads.items(), key=lambda kv: kv[0].lower()))
        if not items:
            return np.array([], dtype=np.float32)

        def to_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().contiguous().view(-1).numpy()

        first = to_np(items[0][1])
        flat = first
        for _, tensor in items[1:]:
            arr = to_np(tensor)
            flat = np.append(flat, -arr / lr)
        return flat
