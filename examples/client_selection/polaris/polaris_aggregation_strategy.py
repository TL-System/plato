"""
Polaris aggregation strategy.

Tracks gradient bounds for unexplored clients to support Polaris client
selection.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from plato.servers.strategies.aggregation.fedavg import FedAvgAggregationStrategy
from plato.servers.strategies.base import ServerContext


class PolarisAggregationStrategy(FedAvgAggregationStrategy):
    """Aggregate deltas while tracking gradient bounds for Polaris."""

    def __init__(
        self,
        alpha: float = 10.0,
        initial_gradient_bound: float = 0.5,
        initial_staleness: float = 0.01,
    ):
        super().__init__()
        self.alpha = alpha
        self.initial_gradient_bound = initial_gradient_bound
        self.initial_staleness = initial_staleness
        self.total_clients = 0
        self.squared_deltas_current_round: Optional[np.ndarray] = None
        self.unexplored_clients: Optional[Set[int]] = None

    def setup(self, context: ServerContext) -> None:
        super().setup(context)

        self.total_clients = context.total_clients
        self.squared_deltas_current_round = np.zeros(self.total_clients)
        self.unexplored_clients = set(range(self.total_clients))

        polaris_state = context.state.setdefault("polaris", {})
        polaris_state.setdefault(
            "local_gradient_bounds",
            np.full(self.total_clients, self.initial_gradient_bound, dtype=float),
        )
        polaris_state.setdefault(
            "local_stalenesses",
            np.full(self.total_clients, self.initial_staleness, dtype=float),
        )
        polaris_state.setdefault(
            "aggregation_weights",
            np.full(self.total_clients, 1.0 / max(1, self.total_clients), dtype=float),
        )
        polaris_state.setdefault(
            "squared_deltas_current_round",
            np.zeros(self.total_clients, dtype=float),
        )

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        avg_update = await super().aggregate_deltas(updates, deltas_received, context)

        if not updates or not deltas_received:
            return avg_update

        polaris_state = context.state.setdefault("polaris", {})
        local_gradient_bounds = polaris_state["local_gradient_bounds"]

        self.squared_deltas_current_round = np.zeros(self.total_clients)
        sum_deltas_current_round = 0.0
        deltas_counter = 0

        for update, delta in zip(updates, deltas_received):
            client_index = update.client_id - 1
            squared_delta = 0.0

            for layer_name, value in delta.items():
                if "conv" not in layer_name:
                    continue

                tensor_value = value
                if isinstance(tensor_value, torch.Tensor):
                    tensor_value = tensor_value.detach().cpu().numpy()
                squared_delta += float(np.sum(np.square(tensor_value)))

            norm_delta = float(np.sqrt(max(squared_delta, 0.0)))
            self.squared_deltas_current_round[client_index] = norm_delta

            if (
                self.unexplored_clients is not None
                and client_index in self.unexplored_clients
            ):
                self.unexplored_clients.remove(client_index)

            sum_deltas_current_round += norm_delta
            deltas_counter += 1

        if deltas_counter > 0:
            avg_deltas_current_round = sum_deltas_current_round / deltas_counter
            expect_deltas = self.alpha * avg_deltas_current_round

            if self.unexplored_clients:
                for client_index in self.unexplored_clients:
                    self.squared_deltas_current_round[client_index] = expect_deltas

        for idx, bound in enumerate(self.squared_deltas_current_round):
            if bound != 0:
                local_gradient_bounds[idx] = bound

        polaris_state["squared_deltas_current_round"] = (
            self.squared_deltas_current_round.copy()
        )

        return avg_update
