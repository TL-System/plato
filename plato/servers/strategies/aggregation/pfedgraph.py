"""
pFedGraph aggregation strategy.

Implements collaboration-graph-based aggregation with simplex projection.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import torch

from plato.servers.strategies.base import AggregationStrategy, ServerContext


def _project_to_simplex(vector: torch.Tensor) -> torch.Tensor:
    """
    Project a vector onto the probability simplex.

    Returns the closest non-negative vector that sums to 1. Uses the algorithm
    from: Wang & Carreira-Perpinan (2013).
    """
    if vector.numel() == 1:
        return torch.ones_like(vector)

    sorted_vec, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(sorted_vec, dim=0) - 1.0
    indices = torch.arange(
        1, vector.numel() + 1, device=vector.device, dtype=vector.dtype
    )
    condition = sorted_vec - cumulative / indices > 0
    if not torch.any(condition):
        theta = cumulative[-1] / vector.numel()
    else:
        rho = torch.nonzero(condition, as_tuple=False)[-1].item()
        theta = cumulative[rho] / (rho + 1)
    return torch.clamp(vector - theta, min=0.0)


def _flatten_weight_diff(
    weights: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    layer_filters: Sequence[str] | None,
) -> torch.Tensor:
    """Flatten weight differences for selected layers."""
    parts = []
    for name, tensor in weights.items():
        if layer_filters and not any(
            filter_key in name for filter_key in layer_filters
        ):
            continue
        parts.append((tensor - reference[name]).reshape(-1))
    if not parts:
        for name, tensor in weights.items():
            parts.append((tensor - reference[name]).reshape(-1))
    return torch.cat(parts)


class PFedGraphAggregationStrategy(AggregationStrategy):
    """
    Aggregation strategy for pFedGraph.

    This strategy:
    - Updates the collaboration graph based on cosine similarity of model updates
    - Aggregates client weights using graph-based coefficients
    - Stores per-client aggregated weights on the server
    """

    def __init__(
        self,
        *,
        alpha: float = 0.8,
        similarity_metric: str = "all",
        similarity_layers: Sequence[str] | None = None,
        eps: float = 1e-12,
    ):
        if alpha <= 0:
            raise ValueError("alpha must be positive for pFedGraph.")

        self.alpha = alpha
        self.similarity_metric = similarity_metric
        self.similarity_layers = list(similarity_layers) if similarity_layers else None
        self.eps = eps
        self.graph_matrix: torch.Tensor | None = None
        self.initial_weights: dict[str, torch.Tensor] | None = None
        self.total_clients = 0

    def setup(self, context: ServerContext) -> None:
        server = getattr(context, "server", None)
        if server is not None:
            self.total_clients = getattr(server, "total_clients", 0)

        algorithm = getattr(context, "algorithm", None)
        if algorithm is not None and hasattr(algorithm, "extract_weights"):
            baseline = algorithm.extract_weights()
            self.initial_weights = {
                name: tensor.detach().cpu().clone()
                for name, tensor in baseline.items()
            }

        if self.total_clients > 0:
            self.graph_matrix = self._init_graph_matrix(self.total_clients)

    async def aggregate_deltas(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[dict],
        context: ServerContext,
    ) -> dict:
        """pFedGraph aggregates weights directly; deltas are not supported."""
        raise RuntimeError(
            "pFedGraph aggregation expects weight aggregation, not deltas."
        )

    async def aggregate_weights(
        self,
        updates: list[SimpleNamespace],
        baseline_weights: dict,
        weights_received: list[dict],
        context: ServerContext,
    ) -> dict | None:
        if not updates or not weights_received:
            return dict(baseline_weights)

        server = getattr(context, "server", None)
        if server is None:
            raise AttributeError("pFedGraph aggregation requires server context.")
        if not hasattr(server, "update_client_model"):
            raise AttributeError(
                "pFedGraph aggregation requires server.update_client_model."
            )

        if self.initial_weights is None:
            self.initial_weights = {
                name: tensor.detach().cpu().clone()
                for name, tensor in baseline_weights.items()
            }

        if self.graph_matrix is None:
            total_clients = (
                getattr(server, "total_clients", 0) or self.total_clients or 0
            )
            if total_clients <= 0:
                total_clients = max(update.client_id for update in updates) + 1
            self.graph_matrix = self._init_graph_matrix(total_clients)
            self.total_clients = total_clients

        selected_ids = [update.client_id for update in updates]
        total_samples = sum(update.report.num_samples for update in updates)
        if total_samples <= 0:
            sample_weights = torch.full(
                (len(updates),),
                1.0 / len(updates),
                dtype=torch.float32,
            )
        else:
            sample_weights = torch.tensor(
                [update.report.num_samples / total_samples for update in updates],
                dtype=torch.float32,
            )

        weights_cpu = [
            {name: tensor.detach().cpu() for name, tensor in weights.items()}
            for weights in weights_received
        ]

        diff_matrix = self._compute_difference_matrix(weights_cpu)
        self._update_graph_rows(selected_ids, diff_matrix, sample_weights)

        aggregated_weights = self._aggregate_client_weights(weights_cpu, selected_ids)
        server.update_client_model(aggregated_weights, updates)

        server.total_samples = total_samples

        global_weights = self._aggregate_global_weights(
            aggregated_weights, sample_weights, baseline_weights
        )
        return global_weights

    def _init_graph_matrix(self, total_clients: int) -> torch.Tensor:
        if total_clients <= 1:
            return torch.ones((total_clients, total_clients), dtype=torch.float32)

        graph = torch.full(
            (total_clients, total_clients),
            1.0 / (total_clients - 1),
            dtype=torch.float32,
        )
        graph[torch.arange(total_clients), torch.arange(total_clients)] = 0.0
        return graph

    def _resolve_similarity_layers(self) -> Sequence[str] | None:
        if self.similarity_layers:
            return self.similarity_layers
        if self.similarity_metric == "fc":
            return ["fc"]
        return None

    def _compute_difference_matrix(
        self, weights_cpu: list[dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        initial_weights = self.initial_weights
        if initial_weights is None:
            raise RuntimeError("pFedGraph initial weights have not been initialised.")

        layer_filters = self._resolve_similarity_layers()
        flat_diffs = [
            _flatten_weight_diff(weights, initial_weights, layer_filters)
            for weights in weights_cpu
        ]
        num_clients = len(flat_diffs)
        diff_matrix = torch.zeros((num_clients, num_clients), dtype=torch.float32)

        for i in range(num_clients):
            for j in range(i, num_clients):
                vec_i = flat_diffs[i]
                vec_j = flat_diffs[j]
                denom = (vec_i.norm() * vec_j.norm()).clamp_min(self.eps)
                cos_sim = torch.dot(vec_i, vec_j) / denom
                diff = -cos_sim
                if diff < -0.9:
                    diff = torch.tensor(-1.0)
                diff_matrix[i, j] = diff
                diff_matrix[j, i] = diff

        return diff_matrix

    def _update_graph_rows(
        self,
        selected_ids: list[int],
        diff_matrix: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> None:
        graph = self.graph_matrix
        if graph is None:
            raise RuntimeError("pFedGraph graph matrix has not been initialised.")

        for row_idx, client_id in enumerate(selected_ids):
            difference_vector = diff_matrix[row_idx]
            candidate = sample_weights - difference_vector / (2.0 * self.alpha)
            projected = _project_to_simplex(candidate)

            graph[client_id, :] = 0.0
            graph[client_id, selected_ids] = projected

    def _aggregate_client_weights(
        self,
        weights_cpu: list[dict[str, torch.Tensor]],
        selected_ids: list[int],
    ) -> list[dict[str, torch.Tensor]]:
        graph = self.graph_matrix
        if graph is None:
            raise RuntimeError("pFedGraph graph matrix has not been initialised.")

        aggregated_weights: list[dict[str, torch.Tensor]] = []
        for client_id in selected_ids:
            row = graph[client_id, selected_ids]
            aggregated = {
                name: torch.zeros_like(value)
                for name, value in weights_cpu[0].items()
            }

            for neighbor_idx, weight in enumerate(row):
                neighbor_weights = weights_cpu[neighbor_idx]
                for name in aggregated:
                    aggregated[name] += neighbor_weights[name] * weight

            aggregated_weights.append(aggregated)

        return aggregated_weights

    def _aggregate_global_weights(
        self,
        aggregated_weights: list[dict[str, torch.Tensor]],
        sample_weights: torch.Tensor,
        baseline_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        global_weights = {
            name: torch.zeros_like(tensor) for name, tensor in baseline_weights.items()
        }

        for idx, client_weights in enumerate(aggregated_weights):
            weight = float(sample_weights[idx])
            for name, tensor in client_weights.items():
                global_weights[name] += tensor.to(
                    dtype=baseline_weights[name].dtype,
                    device=baseline_weights[name].device,
                ) * weight

        return global_weights


__all__ = ["PFedGraphAggregationStrategy"]
