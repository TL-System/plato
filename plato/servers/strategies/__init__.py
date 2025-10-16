"""
Server strategies for composable federated learning.

This package provides strategy pattern implementations for server-side
FL operations, including aggregation and client selection.

Example:
    >>> from plato.servers import fedavg
    >>> from plato.servers.strategies import (
    ...     FedNovaAggregationStrategy,
    ...     RandomSelectionStrategy
    ... )
    >>>
    >>> server = fedavg.Server(
    ...     aggregation_strategy=FedNovaAggregationStrategy(),
    ...     client_selection_strategy=RandomSelectionStrategy()
    ... )
"""

# Base classes and context
# Aggregation strategies
from plato.servers.strategies.aggregation import (
    FedAsyncAggregationStrategy,
    FedAvgAggregationStrategy,
    FedBuffAggregationStrategy,
    FedNovaAggregationStrategy,
)
from plato.servers.strategies.base import (
    AggregationStrategy,
    ClientSelectionStrategy,
    ServerContext,
    ServerStrategy,
)

# Client selection strategies
from plato.servers.strategies.client_selection import RandomSelectionStrategy
from plato.servers.strategies.mpc import (
    MPCAdditiveAggregationStrategy,
    MPCRoundSelectionStrategy,
    MPCShamirAggregationStrategy,
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
    "FedBuffAggregationStrategy",
    "MPCAdditiveAggregationStrategy",
    "MPCShamirAggregationStrategy",
    # Client selection strategies
    "RandomSelectionStrategy",
    "MPCRoundSelectionStrategy",
]
