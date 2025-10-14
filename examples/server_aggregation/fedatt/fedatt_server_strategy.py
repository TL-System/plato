"""
Server aggregation using FedAtt with strategy pattern.

Reference:

S. Ji, S. Pan, G. Long, X. Li, J. Jiang, Z. Huang. "Learning Private Neural Language Modeling
with Attentive Aggregation," in Proc. International Joint Conference on Neural Networks (IJCNN),
2019.

https://arxiv.org/abs/1812.07108
"""

from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedAttAggregationStrategy(AggregationStrategy):
    """FedAtt aggregation strategy using attentive aggregation with norm-based attention."""

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """This method is not used; FedAtt aggregates weights directly."""
        raise NotImplementedError(
            "FedAtt uses aggregate_weights instead of aggregate_deltas"
        )

    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Optional[Dict]:
        """Perform attentive aggregation with the attention mechanism."""
        # Compute weight deltas
        deltas_received = []
        for weight in weights_received:
            delta = OrderedDict()
            for name, current_weight in baseline_weights.items():
                delta[name] = weight[name] - current_weight
            deltas_received.append(delta)

        att_update = {
            name: context.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        atts = OrderedDict()

        for name in baseline_weights.keys():
            atts[name] = context.trainer.zeros(len(deltas_received))
            for i, update in enumerate(deltas_received):
                # convert potential LongTensor to FloatTensor for linalg.norm
                delta = update[name].type(torch.FloatTensor)
                atts[name][i] = torch.linalg.norm(-delta)

        for name in baseline_weights.keys():
            atts[name] = F.softmax(atts[name], dim=0)

        for name, weight in baseline_weights.items():
            att_weight = context.trainer.zeros(weight.shape)
            for i, update in enumerate(deltas_received):
                delta = update[name]
                delta = delta.float()
                att_weight += torch.mul(-delta, atts[name][i])

            # Step size for aggregation used in FedAtt
            epsilon = (
                Config().algorithm.epsilon
                if hasattr(Config().algorithm, "epsilon")
                else 1.2
            )

            # The magnitude of normal noise in the randomization mechanism
            magnitude = (
                Config().algorithm.magnitude
                if hasattr(Config().algorithm, "magnitude")
                else 0.001
            )

            att_update[name] = -torch.mul(att_weight, epsilon) + torch.mul(
                torch.randn(weight.shape), magnitude
            )

        # Apply the aggregated update to the baseline weights
        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + att_update[name]

        return updated_weights
