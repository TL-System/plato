"""
Server strategies for composable federated learning.

This package provides strategy pattern implementations for server-side
FL operations, including aggregation and client selection.

Example:
    >>> from plato.servers import fedavg
    >>> from plato.servers.strategies import (
    ...     FedNovaAggregationStrategy,
    ...     OortSelectionStrategy
    ... )
    >>>
    >>> server = fedavg.Server(
    ...     aggregation_strategy=FedNovaAggregationStrategy(),
    ...     client_selection_strategy=OortSelectionStrategy()
    ... )
"""

# Base classes and context
from plato.servers.strategies.base import (
    AggregationStrategy,
    ClientSelectionStrategy,
    ServerContext,
    ServerStrategy,
)

# Aggregation strategies
from plato.servers.strategies.aggregation import (
    FedAsyncAggregationStrategy,
    FedAvgAggregationStrategy,
    FedNovaAggregationStrategy,
)

# Client selection strategies
from plato.servers.strategies.client_selection import (
    AFLSelectionStrategy,
    OortSelectionStrategy,
    RandomSelectionStrategy,
)

__all__ = [
    # Base classes
    "ServerContext",
    "ServerStrategy",
    "AggregationStrategy",
    "ClientSelectionStrategy",
    # Aggregation strategies
    "FedAvgAggregationStrategy",
    "FedNovaAggregationStrategy",
    "FedAsyncAggregationStrategy",
    # Client selection strategies
    "RandomSelectionStrategy",
    "OortSelectionStrategy",
    "AFLSelectionStrategy",
]
