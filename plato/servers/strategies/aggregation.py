"""
Default aggregation strategy implementations.

This module provides ready-to-use aggregation strategies for common
federated learning algorithms.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from types import SimpleNamespace

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

    Example:
        >>> strategy = FedAsyncAggregationStrategy(
        ...     mixing_hyperparameter=0.9,
        ...     adaptive_mixing=True,
        ...     staleness_func_type='polynomial',
        ...     staleness_func_params={'a': 0.5}
        ... )
        >>> server = fedavg.Server(aggregation_strategy=strategy)
    """

    def __init__(
        self,
        mixing_hyperparameter: float = 0.9,
        adaptive_mixing: bool = False,
        staleness_func_type: str = "constant",
        staleness_func_params: Optional[Dict] = None,
    ):
        """Initialize FedAsync aggregation strategy."""
        super().__init__()
        self.mixing_hyperparam = mixing_hyperparameter
        self.adaptive_mixing = adaptive_mixing
        self.staleness_func_type = staleness_func_type.lower()
        self.staleness_func_params = staleness_func_params or {}

        # Validate mixing hyperparameter
        if not 0 < self.mixing_hyperparam < 1:
            logging.warning(
                "FedAsync: Mixing hyperparameter should be between 0 and 1 (exclusive). "
                "Got: %s",
                self.mixing_hyperparam,
            )

    def setup(self, context: ServerContext) -> None:
        """Setup and validate configuration."""
        # Try to load from config if not provided
        try:
            if hasattr(Config().server, "mixing_hyperparameter"):
                self.mixing_hyperparam = Config().server.mixing_hyperparameter

            if hasattr(Config().server, "adaptive_mixing"):
                self.adaptive_mixing = Config().server.adaptive_mixing
        except ValueError:
            # Config not initialized, use constructor parameters
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
        """
        Aggregate deltas (fallback implementation).

        FedAsync typically aggregates weights directly, but this method
        provides a fallback for compatibility.
        """
        # Use FedAvg-style aggregation as fallback
        total_samples = sum(update.report.num_samples for update in updates)

        avg_update = {
            name: context.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples

            for name, delta in update.items():
                avg_update[name] += delta * (num_samples / total_samples)

            await asyncio.sleep(0)

        return avg_update

    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """Aggregate weights directly with staleness mixing."""
        # Calculate mixing parameter based on staleness
        client_staleness = updates[0].staleness
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

        # Use algorithm's aggregate_weights with mixing parameter
        return await context.algorithm.aggregate_weights(
            baseline_weights, weights_received, mixing=mixing
        )

    def _staleness_function(self, staleness: int) -> float:
        """Calculate staleness weighting factor."""
        if self.staleness_func_type == "constant":
            return self._constant_function()
        elif self.staleness_func_type == "polynomial":
            a = self.staleness_func_params.get("a", 1.0)
            return self._polynomial_function(staleness, a)
        elif self.staleness_func_type == "hinge":
            a = self.staleness_func_params.get("a", 1.0)
            b = self.staleness_func_params.get("b", 10)
            return self._hinge_function(staleness, a, b)
        else:
            logging.warning(
                "FedAsync: Unknown staleness function type '%s'. Using constant.",
                self.staleness_func_type,
            )
            return self._constant_function()

    @staticmethod
    def _constant_function() -> float:
        """Constant staleness function (no adjustment)."""
        return 1.0

    @staticmethod
    def _polynomial_function(staleness: int, a: float) -> float:
        """
        Polynomial staleness function.

        Args:
            staleness: Number of rounds since client started training
            a: Polynomial exponent parameter

        Returns:
            (staleness + 1)^(-a)
        """
        return (staleness + 1) ** (-a)

    @staticmethod
    def _hinge_function(staleness: int, a: float, b: int) -> float:
        """
        Hinge staleness function.

        Args:
            staleness: Number of rounds since client started training
            a: Slope parameter
            b: Threshold parameter

        Returns:
            1 if staleness <= b, else 1/(a*(staleness-b)+1)
        """
        if staleness <= b:
            return 1.0
        else:
            return 1.0 / (a * (staleness - b) + 1)
