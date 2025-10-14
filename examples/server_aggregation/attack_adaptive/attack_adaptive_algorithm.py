"""
Server aggregation using attack-adaptive aggregation.

Reference:

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

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """The federated learning algorithm for attack-adaptive aggregation, used by the server."""

    async def aggregate_weights(self, baseline_weights, weights_received, **kwargs):
        """Aggregate weight updates from the clients using attack-adaptive aggregation."""
        deltas_received = self.compute_weight_deltas(baseline_weights, weights_received)

        # Performing attack-adaptive aggregation
        att_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        # Calculating attention
        atts = OrderedDict()
        for name, weight in baseline_weights.items():
            atts[name] = self.trainer.zeros(len(deltas_received))
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
            att_weight = self.trainer.zeros(weight.shape)
            for i, update in enumerate(deltas_received):
                delta = update[name]
                att_weight += delta.mul(atts[name][i])
            att_update[name] = att_weight

        return self.update_weights(att_update)
