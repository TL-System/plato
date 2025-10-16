"""
Aggregation strategies package.

Each strategy is defined in its own module for clarity.
"""

from plato.servers.strategies.aggregation.fedasync import FedAsyncAggregationStrategy
from plato.servers.strategies.aggregation.fedavg import FedAvgAggregationStrategy
from plato.servers.strategies.aggregation.fedbuff import FedBuffAggregationStrategy
from plato.servers.strategies.aggregation.fednova import FedNovaAggregationStrategy
from plato.servers.strategies.aggregation.hermes import HermesAggregationStrategy

__all__ = [
    "FedAvgAggregationStrategy",
    "FedBuffAggregationStrategy",
    "FedNovaAggregationStrategy",
    "FedAsyncAggregationStrategy",
    "HermesAggregationStrategy",
]
