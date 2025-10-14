"""
Base strategy interfaces for composable server architecture.

This module defines the core strategy interfaces and ServerContext for
the composition-based server design. Instead of using inheritance to extend
server functionality, strategies are injected as dependencies.

Example:
    >>> from plato.servers import fedavg
    >>> from plato.servers.strategies.aggregation import FedNovaAggregationStrategy
    >>>
    >>> server = fedavg.Server(
    ...     aggregation_strategy=FedNovaAggregationStrategy()
    ... )
"""

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


class ServerContext:
    """
    Shared context passed between server strategies during federated learning.

    The ServerContext acts as a data container that allows strategies to:
    - Access common server state (trainer, algorithm, etc.)
    - Share data between strategies via the `state` dictionary
    - Communicate information across training lifecycle

    Attributes:
        server: Reference to the server instance
        trainer: The server's trainer instance
        algorithm: The aggregation algorithm instance
        current_round: Current federated learning round number
        total_clients: Total number of clients in the system
        clients_per_round: Number of clients selected per round
        updates: List of client update objects in current round
        state: Dictionary for strategies to share arbitrary state data

    Example:
        >>> context = ServerContext()
        >>> context.server = server_instance
        >>> context.trainer = trainer_instance
        >>> context.state['custom_data'] = some_value
    """

    def __init__(self):
        """Initialize server context with default values."""
        self.server = None
        self.trainer = None
        self.algorithm = None
        self.current_round: int = 0
        self.total_clients: int = 0
        self.clients_per_round: int = 0
        self.updates: List[SimpleNamespace] = []
        self.state: Dict[str, Any] = {}

    def __repr__(self) -> str:
        """Return string representation of context."""
        return (
            f"ServerContext(round={self.current_round}, "
            f"clients={self.clients_per_round}/{self.total_clients})"
        )


class ServerStrategy(ABC):
    """
    Base class for all server strategies.

    All strategies inherit from this base class and can implement
    setup/teardown lifecycle methods for initialization and cleanup.

    The strategy pattern allows algorithms to be swapped at runtime
    without changing the server implementation.
    """

    def setup(self, context: ServerContext) -> None:
        """
        Called once during server initialization.

        Use this method to:
        - Initialize strategy state
        - Access server configuration
        - Allocate resources
        - Load saved state from disk

        Args:
            context: Server context with references to server, trainer, etc.
        """
        pass

    def teardown(self, context: ServerContext) -> None:
        """
        Called when server closes.

        Use this method to:
        - Clean up resources
        - Save final state to disk
        - Release memory

        Args:
            context: Server context
        """
        pass


class AggregationStrategy(ServerStrategy):
    """
    Strategy interface for aggregating client model updates.

    Implement this interface to customize aggregation behavior:
    - FedAvg weighted averaging
    - FedNova normalized aggregation
    - FedAsync staleness-aware aggregation
    - Custom aggregation algorithms

    Example:
        >>> class MyAggregationStrategy(AggregationStrategy):
        ...     async def aggregate_deltas(self, updates, deltas_received, context):
        ...         total_samples = sum(u.report.num_samples for u in updates)
        ...         avg_update = {}
        ...         for name, delta in deltas_received[0].items():
        ...             avg_update[name] = context.trainer.zeros(delta.shape)
        ...             for i, update in enumerate(deltas_received):
        ...                 weight = updates[i].report.num_samples / total_samples
        ...                 avg_update[name] += update[name] * weight
        ...         return avg_update
    """

    @abstractmethod
    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """
        Aggregate weight deltas from clients.

        This method is called when aggregating model weight deltas (differences
        between client models and global model). Most FL algorithms aggregate
        deltas in a framework-agnostic fashion.

        Args:
            updates: List of client update objects containing:
                - client_id: ID of the client
                - report: Client report with metadata (num_samples, accuracy, etc.)
                - payload: Model weights or deltas
                - staleness: Number of rounds since client started training
            deltas_received: List of weight delta dictionaries, where each
                dictionary maps layer names to delta tensors
            context: Server context with trainer, algorithm, and state

        Returns:
            Aggregated weight deltas as dictionary mapping layer names to tensors

        Example:
            >>> async def aggregate_deltas(self, updates, deltas_received, context):
            ...     total_samples = sum(u.report.num_samples for u in updates)
            ...     avg = {name: context.trainer.zeros(delta.shape)
            ...            for name, delta in deltas_received[0].items()}
            ...     for i, deltas in enumerate(deltas_received):
            ...         weight = updates[i].report.num_samples / total_samples
            ...         for name, delta in deltas.items():
            ...             avg[name] += delta * weight
            ...     return avg
        """
        pass

    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Optional[Dict]:
        """
        Optional: Aggregate model weights directly instead of deltas.

        Some algorithms (like FedAsync) prefer to aggregate weights directly
        rather than computing deltas first. Override this method to provide
        direct weight aggregation.

        Args:
            updates: List of client update objects
            baseline_weights: Current global model weights as dictionary
            weights_received: List of client weight dictionaries
            context: Server context

        Returns:
            Aggregated weights as dictionary, or None to use aggregate_deltas instead

        Note:
            If this returns None, the server will compute deltas and call
            aggregate_deltas instead. Return a weight dictionary to bypass
            delta computation.

        Example:
            >>> async def aggregate_weights(self, updates, baseline, weights, context):
            ...     # Mix baseline with received weights
            ...     alpha = 0.9
            ...     mixed = {}
            ...     for name in baseline.keys():
            ...         mixed[name] = alpha * baseline[name] + (1-alpha) * weights[0][name]
            ...     return mixed
        """
        return None  # Default: use delta aggregation


class ClientSelectionStrategy(ServerStrategy):
    """
    Strategy interface for selecting clients in each round.

    Implement this interface to customize client selection behavior:
    - Random selection (uniform sampling)
    - Oort utility-based selection
    - AFL (Active Federated Learning)
    - Power-of-choice
    - Custom selection algorithms

    Example:
        >>> class MySelectionStrategy(ClientSelectionStrategy):
        ...     def select_clients(self, clients_pool, clients_count, context):
        ...         # Select clients with highest IDs
        ...         return sorted(clients_pool)[-clients_count:]
    """

    @abstractmethod
    def select_clients(
        self,
        clients_pool: List[int],
        clients_count: int,
        context: ServerContext,
    ) -> List[int]:
        """
        Select a subset of clients for the current round.

        Args:
            clients_pool: List of available client IDs
            clients_count: Number of clients to select
            context: Server context with round information and state

        Returns:
            List of selected client IDs

        Note:
            - Use context.state['prng_state'] to maintain reproducible randomness
            - Store any selection state in context.state for future rounds
            - The method should not modify clients_pool

        Example:
            >>> def select_clients(self, clients_pool, clients_count, context):
            ...     import random
            ...     prng_state = context.state.get('prng_state')
            ...     if prng_state:
            ...         random.setstate(prng_state)
            ...     selected = random.sample(clients_pool, clients_count)
            ...     context.state['prng_state'] = random.getstate()
            ...     return selected
        """
        pass

    def on_clients_selected(
        self, selected_clients: List[int], context: ServerContext
    ) -> None:
        """
        Hook called after clients are selected.

        Use this to:
        - Log selection information
        - Update selection statistics
        - Prepare state for selected clients

        Args:
            selected_clients: List of client IDs that were selected
            context: Server context
        """
        pass

    def on_reports_received(
        self, updates: List[SimpleNamespace], context: ServerContext
    ) -> None:
        """
        Hook called after client reports are received and aggregated.

        Use this to:
        - Update client utilities/valuations
        - Collect performance metrics
        - Adjust selection parameters

        Args:
            updates: List of client update objects with reports
            context: Server context

        Example:
            >>> def on_reports_received(self, updates, context):
            ...     for update in updates:
            ...         # Update utility based on loss improvement
            ...         client_id = update.client_id
            ...         utility = update.report.loss_improvement
            ...         self.client_utilities[client_id] = utility
        """
        pass
