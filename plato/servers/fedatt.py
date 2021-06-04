"""
A federated learning server using FedAtt.

Reference:

Ji et al., "Learning Private Neural Language Modeling with Attentive Aggregation,"
in the Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN).

https://arxiv.org/pdf/1812.07108.pdf
"""
from collections import OrderedDict

from plato.servers import fedavg

import torch
import torch.nn.functional as F


class Server(fedavg.Server):
    """A federated learning server using the FedAtt algorithm."""
    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using FedAtt."""
        # Extract weights from the updates
        weights_received = self.extract_client_updates(updates)

        # Performing attentive aggregating
        att_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        # Extract baseline model weights
        baseline_weights = self.algorithm.extract_weights()

        # Calculate attention
        atts = OrderedDict()
        for name, weight in baseline_weights.items():
            atts[name] = self.trainer.zeros(len(weights_received))
            for i, update in enumerate(weights_received):
                delta = update[name]
                atts[name][i] = torch.linalg.norm(weight - delta)
            atts[name] = F.softmax(atts[name], dim=0)

        for name, weight in baseline_weights.items():
            att_weight = self.trainer.zeros(weight.shape)
            for i, update in enumerate(weights_received):
                delta = update[name]
                att_weight += (weight - delta).mul(atts[name][i])

            att_update[name] = -att_weight.mul(1.2)

        # Load updated weights into model
        self.updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            self.updated_weights[name] = weight + att_update[name]
