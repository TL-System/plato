"""
A federated learning server using the algorithm proposed in the following unpublished
manuscript:

Ching Pui Wan, Qifeng Chen, "Robust Federated Learning with Attack-Adaptive Aggregation"
Unpublished
(https://arxiv.org/pdf/2102.05257.pdf)

Comparison to FedAtt, instead of using norm distance, this algorithm uses cosine
similarity between the client and server parameters. It also applies softmax with
temperatures.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the fed_attack_adapt algorithm."""
    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using attack-adaptive aggregation."""
        weights_received = self.extract_client_updates(updates)

        # Performing attack-adaptive aggregation
        att_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        # Extracting baseline model weights
        baseline_weights = self.algorithm.extract_weights()

        # Calculating attention
        atts = OrderedDict()
        for name, weight in baseline_weights.items():
            atts[name] = self.trainer.zeros(len(weights_received))
            for i, update in enumerate(weights_received):
                delta = update[name]

                # Calculating the cosine similarity
                cos = torch.nn.CosineSimilarity(dim=0)
                atts[name][i] = cos(torch.flatten(weight),
                                    torch.flatten(delta))

            # scaling factor for the temperature
            scaling_factor = 10
            atts[name] = F.softmax(atts[name] * scaling_factor, dim=0)

        for name, weight in baseline_weights.items():
            att_weight = self.trainer.zeros(weight.shape)
            for i, update in enumerate(weights_received):
                delta = update[name]
                att_weight += delta.mul(atts[name][i])
            att_update[name] = att_weight

        return att_update