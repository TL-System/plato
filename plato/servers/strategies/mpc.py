"""Server strategies supporting MPC-based aggregation."""

from __future__ import annotations

import asyncio
import logging
import math
from fractions import Fraction
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

    def _validate_round(self, updates, baseline_weights, weights_received, context):
        if len(updates) != len(weights_received):
            raise ValueError("MPC report/payload cardinality mismatch.")
        state = self.round_store.load_state()
        current_round = getattr(context, "current_round", state.round_number)
        if state.round_number != current_round:
            raise RuntimeError("MPC round state does not match the aggregation round.")
        clients = [update.client_id for update in updates]
        if len(clients) != len(set(clients)):
            raise ValueError("Duplicate MPC participants in reports.")
        for update, weights in zip(updates, weights_received):
            count = update.report.num_samples
            if not math.isfinite(count) or count < 0:
                raise ValueError("MPC sample count must be finite and nonnegative.")
            if state.client_samples.get(update.client_id) != count:
                raise ValueError(
                    "MPC report sample count differs from the stored count."
                )
            if update.client_id not in state.selected_clients:
                raise ValueError("Report client is not among the MPC participants.")
            if weights.keys() != baseline_weights.keys():
                raise ValueError("MPC weight keys do not match the baseline.")
        if not math.isfinite(sum(update.report.num_samples for update in updates)):
            raise ValueError("MPC total sample count must be finite.")
        return state

    @staticmethod
    def _validate_shapes(weights, baseline_weights, shamir=False):
        for name, tensor in weights.items():
            expected = tuple(baseline_weights[name].shape) + ((2,) if shamir else ())
            if tuple(tensor.shape) != expected or not torch.isfinite(tensor).all():
                raise ValueError("MPC weight shapes or values are invalid.")

    async def _aggregate_scaled_weights(
        self,
        scaled_weights: list[dict[str, torch.Tensor]],
        updates,
        baseline_weights,
        context,
    ) -> dict[str, torch.Tensor]:
        total_samples = sum(update.report.num_samples for update in updates)
        if total_samples == 0:
            LOGGER.warning(
                "No samples reported in MPC round; retaining baseline weights."
            )
            return baseline_weights

        aggregated: dict[str, torch.Tensor] = {}
        for weight_dict in scaled_weights:
            for name, tensor in weight_dict.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(tensor)
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
        state = self._validate_round(
            updates, baseline_weights, weights_received, context
        )
        if {update.client_id for update in updates} != set(state.selected_clients):
            raise ValueError("Additive aggregation requires all selected participants.")
        combined = []
        for update, weights in zip(updates, weights_received):
            self._validate_shapes(weights, baseline_weights)
            client_id = update.client_id
            share = state.additive_shares.get(client_id)
            peers = set(state.selected_clients) - {client_id}
            if state.additive_contributors.get(client_id, set()) != peers:
                raise ValueError("Missing additive shares from selected participants.")
            if share is not None:
                if share.keys() != baseline_weights.keys():
                    raise ValueError("Mismatched additive share keys.")
                self._validate_shapes(share, baseline_weights)
                merged = {name: weights[name] + share[name] for name in weights}
            else:
                merged = weights
            combined.append(merged)

        return await self._aggregate_scaled_weights(
            combined, updates, baseline_weights, context
        )


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
        xs_int = [int(round(val)) for val in xs[:threshold]]
        ys_int = [int(round(val)) for val in ys[:threshold]]

        if len(xs_int) < threshold or len(set(xs_int)) != threshold:
            raise ValueError("Insufficient distinct Shamir coordinates.")
        accumulator = Fraction(0, 1)
        for i in range(threshold):
            term = Fraction(ys_int[i], 1)
            for j in range(threshold):
                if i == j:
                    continue
                term *= Fraction(-xs_int[j], xs_int[i] - xs_int[j])
            accumulator += term
        return float(accumulator) / self.SCALING_FACTOR

    def _decrypt_tensor(
        self, tensors: torch.Tensor, threshold: int | None = None
    ) -> torch.Tensor:
        num_participants = tensors.size(0)
        threshold = threshold if threshold is not None else max(num_participants - 2, 1)
        if not 1 <= threshold <= num_participants or tensors.shape[-1] != 2:
            raise ValueError("Invalid Shamir threshold or coordinate shape.")
        if not torch.isfinite(tensors).all():
            raise ValueError("Shamir coordinates must be finite.")

        num_weights = int(tensors.numel() / (num_participants * 2))
        coords = tensors.view(num_participants, num_weights, 2)
        secret = torch.zeros([num_weights], dtype=torch.float64)

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
        state = self._validate_round(
            updates, baseline_weights, weights_received, context
        )
        selected = state.selected_clients
        combined = []
        threshold = self.threshold

        client_index = {client_id: idx for idx, client_id in enumerate(selected)}

        for update, weights in zip(updates, weights_received):
            self._validate_shapes(weights, baseline_weights, shamir=True)
            target = update.client_id
            idx = client_index.get(target)
            if idx is None:
                raise RuntimeError(f"Client {target} not present in MPC round state.")

            reconstructed: dict[str, torch.Tensor] = {}
            for name, tensor in weights.items():
                tensor_size = list(tensor.size())
                tensor_size.insert(0, len(selected))
                stacked = torch.zeros(tensor_size, dtype=tensor.dtype)
                stacked[0] = tensor
                insert = 1
                for peer in selected:
                    if peer == target:
                        continue
                    # Reconstruct the target sender's polynomial at each recipient.
                    pair_share = state.pairwise_shares.get((peer, target))
                    if pair_share is None:
                        raise RuntimeError(
                            f"Missing Shamir share from={target}, to={peer}."
                        )
                    if pair_share.keys() != baseline_weights.keys():
                        raise ValueError("Mismatched Shamir share keys.")
                    self._validate_shapes(pair_share, baseline_weights, shamir=True)
                    stacked[insert] = pair_share[name]
                    insert += 1

                reconstructed[name] = self._decrypt_tensor(stacked, threshold)
                if baseline_weights[name].is_floating_point():
                    reconstructed[name] = reconstructed[name].to(
                        baseline_weights[name].dtype
                    )

            combined.append(reconstructed)

        return await self._aggregate_scaled_weights(
            combined, updates, baseline_weights, context
        )
