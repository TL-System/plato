"""
Polaris client selection strategy.
"""

from __future__ import annotations

import logging
import random
from types import SimpleNamespace
from typing import List

import numpy as np

try:
    import mosek  # type: ignore
    from cvxopt import log, matrix, solvers, sparse
except ImportError:  # pragma: no cover - optional dependency
    mosek = None  # type: ignore
    log = matrix = solvers = sparse = None  # type: ignore

from plato.config import Config
from plato.servers.strategies.base import ClientSelectionStrategy, ServerContext


class PolarisSelectionStrategy(ClientSelectionStrategy):
    """Optimise sampling probabilities using geometric programming."""

    def __init__(self, beta: float = 1.0, staleness_weight: float = 1.0):
        super().__init__()
        self.beta = beta
        self.staleness_weight = staleness_weight
        self.total_clients = 0
        self.local_gradient_bounds: np.ndarray | None = None
        self.local_stalenesses: np.ndarray | None = None
        self.aggregation_weights: np.ndarray | None = None

    def setup(self, context: ServerContext) -> None:
        """Initialise Polaris selection state."""
        self._ensure_solver_available()

        try:
            server_cfg = Config().server
            if hasattr(server_cfg, "polaris_beta"):
                self.beta = server_cfg.polaris_beta
            if hasattr(server_cfg, "polaris_staleness_weight"):
                self.staleness_weight = server_cfg.polaris_staleness_weight
        except ValueError:
            pass

        self.total_clients = context.total_clients
        polaris_state = context.state.setdefault("polaris", {})

        polaris_state.setdefault(
            "local_gradient_bounds",
            np.full(self.total_clients, 0.5, dtype=float),
        )
        polaris_state.setdefault(
            "local_stalenesses",
            np.full(self.total_clients, 0.01, dtype=float),
        )
        polaris_state.setdefault(
            "aggregation_weights",
            np.full(self.total_clients, 1.0 / max(1, self.total_clients), dtype=float),
        )

        self.local_gradient_bounds = polaris_state["local_gradient_bounds"]
        self.local_stalenesses = polaris_state["local_stalenesses"]
        self.aggregation_weights = polaris_state["aggregation_weights"]

    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        """Sample clients according to the optimised probability distribution."""
        self._ensure_solver_available()

        assert clients_count <= len(clients_pool)

        polaris_state = context.state.setdefault("polaris", {})
        self.local_gradient_bounds = polaris_state.get("local_gradient_bounds")
        self.local_stalenesses = polaris_state.get("local_stalenesses")
        self.aggregation_weights = polaris_state.get("aggregation_weights")

        if (
            self.local_gradient_bounds is None
            or self.local_stalenesses is None
            or self.aggregation_weights is None
        ):
            raise RuntimeError("PolarisSelection: required state arrays are missing.")

        probabilities = self._calculate_selection_probability(clients_pool)

        prng_state = context.state.get("prng_state")
        if prng_state:
            random.setstate(prng_state)

        selected_clients = np.random.choice(
            clients_pool, clients_count, replace=False, p=probabilities
        ).tolist()

        context.state["prng_state"] = random.getstate()

        logging.info("[Server] Polaris selected clients: %s", selected_clients)
        return selected_clients

    def on_reports_received(
        self, updates: List[SimpleNamespace], context: ServerContext
    ) -> None:
        """Update staleness and aggregation weights from received reports."""
        if not updates:
            return

        polaris_state = context.state.setdefault("polaris", {})
        self.local_stalenesses = polaris_state.setdefault(
            "local_stalenesses",
            np.full(self.total_clients, 0.01, dtype=float),
        )
        self.aggregation_weights = polaris_state.setdefault(
            "aggregation_weights",
            np.full(self.total_clients, 1.0 / max(1, self.total_clients), dtype=float),
        )

        total_samples = sum(update.report.num_samples for update in updates)
        for update in updates:
            client_index = update.client_id - 1
            staleness = getattr(update, "staleness", 0.0)
            self.local_stalenesses[client_index] = staleness + 0.1

            if total_samples > 0:
                self.aggregation_weights[client_index] = (
                    update.report.num_samples / total_samples
                )

    def _ensure_solver_available(self) -> None:
        """Ensure optional optimisation dependencies are available."""
        if (
            mosek is None
            or solvers is None
            or matrix is None
            or sparse is None
            or log is None
        ):
            raise ImportError("PolarisSelectionStrategy requires 'mosek' and 'cvxopt'.")

    def _calculate_selection_probability(self, clients_pool: List[int]) -> np.ndarray:
        """Solve the geometric program defining Polaris sampling probabilities."""
        zero_indexed = [client_id - 1 for client_id in clients_pool]
        num_clients = len(zero_indexed)

        agg_weights = self.aggregation_weights[zero_indexed]
        gradient_bounds = self.local_gradient_bounds[zero_indexed]
        staleness = np.square(self.local_stalenesses[zero_indexed])

        agg_weight_square = np.square(agg_weights)
        gradient_bound_square = np.square(gradient_bounds)

        f1_params = matrix(
            self.beta * np.multiply(agg_weight_square, gradient_bound_square)
        )

        f2_temp = np.multiply(staleness, gradient_bounds)
        f2_params = matrix(
            self.staleness_weight * np.multiply(agg_weight_square, f2_temp)
        )

        f1 = matrix(-1.0 * np.eye(num_clients))
        f2 = matrix(np.eye(num_clients))
        F = sparse([[f1, f2]])

        g = log(matrix(sparse([[f1_params, f2_params]])))

        K = [2 * num_clients]
        G = matrix(-1.0 * np.eye(num_clients))
        h = matrix(np.zeros((num_clients, 1)))

        A = matrix([[1.0]])
        if num_clients > 1:
            A1 = matrix([[1.0]])
            for _ in range(num_clients - 1):
                A = sparse([[A], [A1]])
        b = matrix([1.0])

        solvers.options["maxiters"] = 500
        solution = solvers.gp(
            K, F, g, G, h, A, b, solver="mosek" if mosek is not None else None
        )["x"]

        probabilities = np.array(solution, dtype=float).reshape(-1)
        probabilities = probabilities / probabilities.sum()
        return probabilities
