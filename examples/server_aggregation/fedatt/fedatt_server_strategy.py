"""
Server aggregation using FedAtt with strategy pattern.

Reference:

S. Ji, S. Pan, G. Long, X. Li, J. Jiang, Z. Huang. "Learning Private Neural Language Modeling
with Attentive Aggregation," in Proc. International Joint Conference on Neural Networks (IJCNN),
2019.

https://arxiv.org/abs/1812.07108
"""

from types import SimpleNamespace
from typing import Dict, List, Optional

from plato.config import Config
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedAttAggregationStrategy(AggregationStrategy):
    """FedAtt aggregation strategy delegating tensor operations to the algorithm."""

    async def aggregate_deltas(
        self,
        updates: list[SimpleNamespace],
        deltas_received: list[dict],
        context: ServerContext,
    ) -> dict:
        """This method is not used; FedAtt aggregates weights directly."""
        raise NotImplementedError(
            "FedAtt uses aggregate_weights instead of aggregate_deltas"
        )

    async def aggregate_weights(
        self,
        updates: list[SimpleNamespace],
        baseline_weights: dict,
        weights_received: list[dict],
        context: ServerContext,
    ) -> dict | None:
        """Perform attentive aggregation by delegating to the algorithm."""
        algorithm = getattr(context, "algorithm", None)
        if algorithm is None:
            raise RuntimeError("FedAtt requires an algorithm instance in context.")

        epsilon = (
            Config().algorithm.epsilon
            if hasattr(Config().algorithm, "epsilon")
            else 1.2
        )
        magnitude = (
            Config().algorithm.magnitude
            if hasattr(Config().algorithm, "magnitude")
            else 0.001
        )

        return algorithm.attentive_aggregate_weights(
            baseline_weights,
            weights_received,
            epsilon=epsilon,
            magnitude=magnitude,
        )
