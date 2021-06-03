"""
A federated learning server using fed_attack_adapt.

Reference:

Ching Pui Wan, Qifeng Chen, "Robust Federated Learning with Attack-Adaptive Aggregation"
(https://arxiv.org/pdf/2102.05257.pdf)

Comparison to FedAtt:

Instead of using norm distance, fed_attack_adapt uses cosine similarity between client parameters and server parameters.

It also applies softmax with temperatures.

So actually the only difference is how they calculate the attention score.
"""
from collections import OrderedDict

from plato.servers import fedavg

import logging
import torch
import torch.nn.functional as F
import numpy as np


class Server(fedavg.Server):
    """A federated learning server using the fed_attack_adapt algorithm."""
    async def federated_averaging(self, reports):
        """Aggregate weight updates from the clients using fed_attack_adapt."""
        # Extracting updates from the reports
        updates = self.extract_client_updates(reports)

        # Performing attentive aggregating
        att_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in updates[0].items()
        }

        # Extract baseline model weights
        baseline_weights = self.algorithm.extract_weights()

        # Calculate attention
        atts = OrderedDict()
        for name, weight in baseline_weights.items():
            atts[name] = self.trainer.zeros(len(updates))
            for i, update in enumerate(updates):
                delta = update[name]
                cos = torch.nn.CosineSimilarity(dim=0)
                # cosine similarity
                atts[name][i] = cos(torch.flatten(weight), torch.flatten(delta))
            c = 10 # scaling factor for temperature
            atts[name] = F.softmax(atts[name]*c, dim=0)

        for name, weight in baseline_weights.items():
            att_weight = self.trainer.zeros(weight.shape)
            for i, update in enumerate(updates):
                delta = update[name]
                att_weight += delta.mul(atts[name][i])
            att_update[name] = att_weight
        # Load updated weights into model
        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = att_update[name]

        self.updated_weights = updated_weights
