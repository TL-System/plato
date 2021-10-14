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

from plato.servers import fedavg


class Server(fedavg.Server):
    """ A federated learning server using the FedAtt algorithm. """
    def __init__(self):
        super().__init__()

        # epsilon: step size for aggregation
        self.epsilon = 1.0

        # dp: the magnitude of normal noise in the randomization mechanism
        self.dp = 0.001

    async def federated_averaging(self, updates):
        """ Aggregate weight updates from the clients using FedAtt. """
        # Extract weights from the updates
        weights_received = self.extract_client_updates(updates)

        # Extract baseline model weights
        baseline_weights = self.algorithm.extract_weights()

        # Update server weights
        update = self.avg_att(baseline_weights, weights_received)

        return update

    def avg_att(self, baseline_weights, weights_received):
        """ Perform attentive aggregation with the attention mechanism. """
        att_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        atts = OrderedDict()
        for name, weight in baseline_weights.items():
            atts[name] = self.trainer.zeros(len(weights_received))
            for i, update in enumerate(weights_received):
                delta = update[name]
                atts[name][i] = torch.linalg.norm(delta)

        for name in baseline_weights.keys():
            atts[name] = F.softmax(atts[name], dim=0)

        for name, weight in baseline_weights.items():
            att_weight = self.trainer.zeros(weight.shape)
            for i, update in enumerate(weights_received):
                delta = update[name]
                att_weight += torch.mul(delta, atts[name][i])

            att_update[name] = torch.mul(att_weight, self.epsilon) + torch.mul(
                torch.randn(weight.shape), self.dp)

        return att_update
