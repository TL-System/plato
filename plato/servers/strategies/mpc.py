"""Server strategies supporting MPC-based aggregation."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from plato.mpc import RoundInfoStore
from plato.servers.strategies.aggregation import FedAvgAggregationStrategy
from plato.servers.strategies.client_selection import RandomSelectionStrategy

LOGGER = logging.getLogger(__name__)


class MPCRoundSelectionStrategy(RandomSelectionStrategy):
    """Random selection augmented with MPC round-state initialisation."""

    def __init__(self, round_store: RoundInfoStore):
        super().__init__()
        self.round_store = round_store

    def select_clients(self, clients_pool, clients_count, context):  # noqa: D401
        selected = super().select_clients(clients_pool, clients_count, context)
        self.round_store.initialise_round(context.current_round, selected)
        return selected


class MPCBaseAggregationStrategy(FedAvgAggregationStrategy):
    """Shared helpers for MPC aggregation strategies."""

    def __init__(self, round_store: RoundInfoStore, debug_artifacts: bool = False):
        super().__init__()
        self.round_store = round_store
        self.debug_artifacts = debug_artifacts

    async def _aggregate_scaled_weights(
        self,
        scaled_weights: List[Dict[str, torch.Tensor]],
        updates,
        baseline_weights,
        context,
    ) -> Dict[str, torch.Tensor]:
        total_samples = sum(update.report.num_samples for update in updates)
        if total_samples == 0:
            LOGGER.warning(
                "No samples reported in MPC round; retaining baseline weights."
            )
            return baseline_weights

        aggregated: Dict[str, torch.Tensor] = {}
        for weight_dict in scaled_weights:
            for name, tensor in weight_dict.items():
                if name not in aggregated:
                    aggregated[name] = context.trainer.zeros(tensor.shape)
                aggregated[name] += tensor
            await asyncio.sleep(0)

        for name in aggregated:
            aggregated[name] /= total_samples

        return aggregated


class MPCAdditiveAggregationStrategy(MPCBaseAggregationStrategy):
    """Reconstructs additive-secret-shared payloads before aggregation."""

    async def aggregate_weights(
        self, updates, baseline_weights, weights_received, context
    ):
        state = self.round_store.load_state()
        combined = []
        for update, weights in zip(updates, weights_received):
            client_id = update.client_id
            share = state.additive_shares.get(client_id)
            if share is not None:
                merged = {name: weights[name] + share[name] for name in weights}
            else:
                merged = weights
            combined.append(merged)

        return await self._aggregate_scaled_weights(
            combined, updates, baseline_weights, context
        )


@dataclass
class _Fraction:
    num: float
    den: float

    def reduce(self) -> None:
        gcd = math.gcd(int(self.num), int(self.den))
        if gcd:
            self.num = int(self.num / gcd)
            self.den = int(self.den / gcd)

    def multiply(self, other: "_Fraction") -> "_Fraction":
        result = _Fraction(self.num * other.num, self.den * other.den)
        result.reduce()
        return result

    def add(self, other: "_Fraction") -> "_Fraction":
        result = _Fraction(
            self.num * other.den + self.den * other.num,
            self.den * other.den,
        )
        result.reduce()
        return result


class MPCShamirAggregationStrategy(MPCBaseAggregationStrategy):
    """Recovers plaintext updates from Shamir-secret-shared payloads."""

    SCALING_FACTOR = 1_000_000

    def __init__(
        self,
        round_store: RoundInfoStore,
        debug_artifacts: bool = False,
        threshold: int | None = None,
    ):
        super().__init__(round_store, debug_artifacts)
        self.threshold = threshold

    def _recover_secret(self, xs: np.ndarray, ys: np.ndarray, threshold: int) -> float:
        accumulator = _Fraction(0, 1)
        for i in range(threshold):
            term = _Fraction(ys[i], 1)
            for j in range(threshold):
                if i == j:
                    continue
                term = term.multiply(_Fraction(-xs[j], xs[i] - xs[j]))
            accumulator = accumulator.add(term)
        return (accumulator.num / accumulator.den) / self.SCALING_FACTOR

    def _decrypt_tensor(
        self, tensors: torch.Tensor, threshold: int | None = None
    ) -> torch.Tensor:
        num_participants = tensors.size(0)
        threshold = threshold or max(num_participants - 2, 1)

        num_weights = int(tensors.numel() / (num_participants * 2))
        coords = tensors.view(num_participants, num_weights, 2)
        secret = torch.zeros([num_weights], dtype=torch.float32)

        for idx in range(num_weights):
            points = coords[:, idx, :].cpu().numpy()
            xs = []
            ys = []
            seen = set()
            for x_val, y_val in points:
                int_x = int(round(x_val))
                if int_x not in seen:
                    xs.append(int_x)
                    ys.append(y_val)
                    seen.add(int_x)
                if len(xs) == threshold:
                    break

            xs_arr = np.array(xs)
            ys_arr = np.array(ys)
            secret[idx] = self._recover_secret(xs_arr, ys_arr, threshold)

        output_shape = list(tensors.size())
        output_shape.pop(0)
        output_shape.pop(-1)
        return secret.view(output_shape)

    async def aggregate_weights(
        self, updates, baseline_weights, weights_received, context
    ):
        state = self.round_store.load_state()
        selected = state.selected_clients
        combined = []
        threshold = self.threshold

        client_index = {client_id: idx for idx, client_id in enumerate(selected)}

        for update, weights in zip(updates, weights_received):
            target = update.client_id
            idx = client_index.get(target)
            if idx is None:
                raise RuntimeError(f"Client {target} not present in MPC round state.")

            reconstructed: Dict[str, torch.Tensor] = {}
            for name, tensor in weights.items():
                tensor_size = list(tensor.size())
                tensor_size.insert(0, len(selected))
                stacked = torch.zeros(tensor_size, dtype=tensor.dtype)
                stacked[0] = tensor
                insert = 1
                for peer in selected:
                    if peer == target:
                        continue
                    pair_share = state.pairwise_shares.get((target, peer))
                    if pair_share is None:
                        raise RuntimeError(
                            f"Missing Shamir share for target={target}, from={peer}."
                        )
                    stacked[insert] = pair_share[name]
                    insert += 1

                reconstructed[name] = self._decrypt_tensor(stacked, threshold)

            combined.append(reconstructed)

        return await self._aggregate_scaled_weights(
            combined, updates, baseline_weights, context
        )
