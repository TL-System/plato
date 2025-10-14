"""
Server aggregation using attack-adaptive aggregation with strategy pattern.

Reference:

Ching Pui Wan, Qifeng Chen, "Robust Federated Learning with Attack-Adaptive Aggregation"
Unpublished
(https://arxiv.org/pdf/2102.05257.pdf)

Comparison to FedAtt, instead of using norm distance, this algorithm uses cosine
similarity between the client and server parameters. It also applies softmax with
temperatures.
"""

from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class AttackAdaptiveAggregationStrategy(AggregationStrategy):
    """Attack-adaptive aggregation strategy using cosine similarity and softmax."""

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """This method is not used; attack-adaptive aggregates weights directly."""
        raise NotImplementedError(
            "Attack-adaptive uses aggregate_weights instead of aggregate_deltas"
        )

    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Optional[Dict]:
        """Aggregate weight updates from the clients using attack-adaptive aggregation."""
        # Compute weight deltas
        deltas_received = []
        for weight in weights_received:
            delta = OrderedDict()
            for name, current_weight in baseline_weights.items():
                delta[name] = weight[name] - current_weight
            deltas_received.append(delta)

        # Performing attack-adaptive aggregation
        att_update = {
            name: context.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        # Calculating attention
        atts = OrderedDict()
        for name, weight in baseline_weights.items():
            atts[name] = context.trainer.zeros(len(deltas_received))
            for i, update in enumerate(deltas_received):
                delta = update[name]

                # Calculating the cosine similarity
                cos = torch.nn.CosineSimilarity(dim=0)
                atts[name][i] = cos(torch.flatten(weight), torch.flatten(delta))

            # Scaling factor for the temperature
            scaling_factor = (
                Config().algorithm.scaling_factor
                if hasattr(Config().algorithm, "scaling_factor")
                else 10
            )
            atts[name] = F.softmax(atts[name] * scaling_factor, dim=0)

        for name, weight in baseline_weights.items():
            att_weight = context.trainer.zeros(weight.shape)
            for i, update in enumerate(deltas_received):
                delta = update[name]
                att_weight += delta.mul(atts[name][i])
            att_update[name] = att_weight

        # Apply the aggregated update to the baseline weights
        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + att_update[name]

        return updated_weights
