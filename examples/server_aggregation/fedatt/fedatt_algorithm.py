"""
A federated learning server using FedAtt.

Reference:

Ji et al., "Learning Private Neural Language Modeling with Attentive Aggregation,"
in the Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN).

https://arxiv.org/abs/1812.07108
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """The federated learning algorithm for FedAtt, used by the server."""

    async def aggregate_weights(self, baseline_weights, weights_received, **kwargs):
        """Perform attentive aggregation with the attention mechanism."""
        deltas_received = self.compute_weight_deltas(baseline_weights, weights_received)

        att_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        atts = OrderedDict()

        for name in baseline_weights.keys():
            atts[name] = self.trainer.zeros(len(deltas_received))
            for i, update in enumerate(deltas_received):
                # convert potential LongTensor to FloatTensor for linalg.norm
                delta = update[name].type(torch.FloatTensor)
                atts[name][i] = torch.linalg.norm(-delta)

        for name in baseline_weights.keys():
            atts[name] = F.softmax(atts[name], dim=0)

        for name, weight in baseline_weights.items():
            att_weight = self.trainer.zeros(weight.shape)
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

        return self.update_weights(att_update)
