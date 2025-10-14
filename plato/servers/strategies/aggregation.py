"""
Default aggregation strategy implementations.

This module provides ready-to-use aggregation strategies for common
federated learning algorithms.
"""

import asyncio
import logging
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

from plato.config import Config
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedAvgAggregationStrategy(AggregationStrategy):
    """
    Standard Federated Averaging aggregation.

    Performs weighted averaging of client deltas based on the number of samples
    each client trained on. This is the most common aggregation method in FL.

    Reference:
        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017.

    Example:
        >>> strategy = FedAvgAggregationStrategy()
        >>> server = fedavg.Server(aggregation_strategy=strategy)
    """

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """Aggregate using weighted average by sample count."""
        # Extract total number of samples
        total_samples = sum(update.report.num_samples for update in updates)

        # Initialize aggregated deltas
        avg_update = {
            name: context.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        # Weighted averaging
        for i, update in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / total_samples)

            # Yield to other async tasks in the server
            await asyncio.sleep(0)

        return avg_update


class FedBuffAggregationStrategy(AggregationStrategy):
    """
    FedBuff aggregation with simple equal-weight averaging.

    Buffers asynchronous updates and applies uniform weighting when aggregating,
    matching the FedBuff algorithm behavior.
    """

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """Aggregate using uniform weights across buffered updates."""
        if not deltas_received:
            return {}

        total_updates = len(deltas_received)
        weight = 1.0 / total_updates if total_updates > 0 else 0.0

        avg_update = {
            name: context.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for delta in deltas_received:
            for name, value in delta.items():
                avg_update[name] += value * weight

            await asyncio.sleep(0)

        return avg_update


class FedNovaAggregationStrategy(AggregationStrategy):
    """
    FedNova aggregation with normalized momentum.

    Addresses the objective inconsistency problem in heterogeneous FL
    by normalizing local updates according to the number of local epochs.

    Reference:
        Wang et al., "Tackling the Objective Inconsistency Problem in
        Heterogeneous Federated Optimization", NeurIPS 2020.

    Example:
        >>> strategy = FedNovaAggregationStrategy()
        >>> server = fedavg.Server(aggregation_strategy=strategy)
    """

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """Aggregate using FedNova normalized averaging."""
        # Extract the total number of samples
        total_samples = sum(update.report.num_samples for update in updates)

        # Extract the number of local epochs (tau_i) from the updates
        local_epochs = [update.report.epochs for update in updates]

        # Initialize aggregated deltas
        avg_update = {
            name: context.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        # Calculate effective tau
        tau_eff = 0
        for i, update in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples
            tau_eff_ = local_epochs[i] * num_samples / total_samples
            tau_eff += tau_eff_

        # Normalized aggregation
        for i, update in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples

            for name, delta in update.items():
                # Apply FedNova normalization
                avg_update[name] += (
                    delta * (num_samples / total_samples) * tau_eff / local_epochs[i]
                )

        return avg_update


class FedAsyncAggregationStrategy(AggregationStrategy):
    """
    FedAsync aggregation with staleness-aware mixing.

    Implements asynchronous federated learning with configurable staleness
    functions to weight client updates based on how stale they are.

    Reference:
        Xie et al., "Asynchronous federated optimization",
        OPT Workshop 2020.

    Args:
        mixing_hyperparameter: Base mixing parameter (0, 1)
        adaptive_mixing: Whether to adjust mixing based on staleness
        staleness_func_type: Type of staleness function ('constant', 'polynomial', 'hinge')
        staleness_func_params: Parameters for the staleness function
    """

    def __init__(
        self,
        mixing_hyperparameter: float = 0.9,
        adaptive_mixing: bool = False,
        staleness_func_type: str = "constant",
        staleness_func_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.mixing_hyperparam = mixing_hyperparameter
        self.adaptive_mixing = adaptive_mixing
        self.staleness_func_type = staleness_func_type.lower()
        self.staleness_func_params = staleness_func_params or {}

        if not 0 < self.mixing_hyperparam < 1:
            logging.warning(
                "FedAsync: Mixing hyperparameter should be between 0 and 1 (exclusive). "
                "Got: %s",
                self.mixing_hyperparam,
            )

    def setup(self, context: ServerContext) -> None:
        try:
            if hasattr(Config().server, "mixing_hyperparameter"):
                self.mixing_hyperparam = Config().server.mixing_hyperparameter
            if hasattr(Config().server, "adaptive_mixing"):
                self.adaptive_mixing = Config().server.adaptive_mixing
        except ValueError:
            pass

        logging.info(
            "FedAsync: Mixing hyperparameter set to %s (adaptive=%s)",
            self.mixing_hyperparam,
            self.adaptive_mixing,
        )

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """Fallback delta aggregation (FedAvg-style)."""
        total_samples = sum(update.report.num_samples for update in updates)

        avg_update = {
            name: context.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, delta in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples
            weight = num_samples / total_samples if total_samples > 0 else 0.0

            for name, value in delta.items():
                avg_update[name] += value * weight

            await asyncio.sleep(0)

        return avg_update

    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """Aggregate weights directly with staleness-aware mixing."""
        if not updates:
            return baseline_weights

        client_staleness = getattr(updates[0], "staleness", 0)
        mixing = self.mixing_hyperparam

        if self.adaptive_mixing:
            staleness_factor = self._staleness_function(client_staleness)
            mixing *= staleness_factor
            logging.debug(
                "FedAsync: Adjusted mixing to %s (staleness=%s, factor=%s)",
                mixing,
                client_staleness,
                staleness_factor,
            )

        return await context.algorithm.aggregate_weights(
            baseline_weights, weights_received, mixing=mixing
        )

    def _staleness_function(self, staleness: int) -> float:
        """Calculate staleness weighting factor."""
        if self.staleness_func_type == "constant":
            return self._constant_function()
        if self.staleness_func_type == "polynomial":
            a = self.staleness_func_params.get("a", 1.0)
            return self._polynomial_function(staleness, a)
        if self.staleness_func_type == "hinge":
            a = self.staleness_func_params.get("a", 1.0)
            b = self.staleness_func_params.get("b", 10)
            return self._hinge_function(staleness, a, b)

        logging.warning(
            "FedAsync: Unknown staleness function type '%s'. Using constant.",
            self.staleness_func_type,
        )
        return self._constant_function()

    @staticmethod
    def _constant_function() -> float:
        return 1.0

    @staticmethod
    def _polynomial_function(staleness: int, a: float) -> float:
        return 1 / (staleness + 1) ** a

    @staticmethod
    def _hinge_function(staleness: int, a: float, b: int) -> float:
        if staleness <= b:
            return 1.0
        return 1 / (a * (staleness - b) + 1)


class PiscesAggregationStrategy(AggregationStrategy):
    """
    Pisces aggregation with staleness-aware weighting.

    Applies a polynomial decay to client updates based on their staleness:
    factor = 1.0 / (staleness + 1) ** staleness_factor
    """

    def __init__(self, staleness_factor: float = 1.0, history_window: int = 5):
        super().__init__()
        self.staleness_factor = staleness_factor
        self.history_window = history_window
        self.client_staleness: Dict[int, List[float]] = {}

    def setup(self, context: ServerContext) -> None:
        try:
            if hasattr(Config().server, "staleness_factor"):
                self.staleness_factor = Config().server.staleness_factor
        except ValueError:
            pass

        total_clients = context.total_clients
        if total_clients:
            self.client_staleness = {
                client_id: [] for client_id in range(1, total_clients + 1)
            }

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        if not updates or not deltas_received:
            return {}

        total_samples = sum(update.report.num_samples for update in updates)
        if total_samples == 0:
            logging.warning("PiscesAggregation: total_samples is 0, returning zeros.")

        avg_update = {
            name: context.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, delta in enumerate(deltas_received):
            client_id = updates[i].client_id
            report = updates[i].report
            num_samples = report.num_samples
            staleness = getattr(updates[i], "staleness", 0.0)

            self.client_staleness.setdefault(client_id, []).append(staleness)
            staleness_factor = self._calculate_staleness_factor(client_id)
            weight = (
                (num_samples / total_samples) * staleness_factor
                if total_samples > 0
                else 0.0
            )

            for name, value in delta.items():
                avg_update[name] += value * weight

            await asyncio.sleep(0)

        return avg_update

    def _calculate_staleness_factor(self, client_id: int) -> float:
        history = self.client_staleness.get(client_id, [])
        if not history:
            return 1.0

        recent_history = history[-self.history_window :]
        staleness = float(np.mean(recent_history))
        return 1.0 / pow(staleness + 1.0, self.staleness_factor)


class PolarisAggregationStrategy(FedAvgAggregationStrategy):
    """
    Polaris aggregation with gradient bound tracking for unexplored clients.

    Computes convolutional gradient norms for reporting clients, estimates
    bounds for unexplored clients, and keeps client-level statistics for the
    selection strategy.
    """

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
        self.unexplored_clients: Optional[set[int]] = None

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
