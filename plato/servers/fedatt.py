"""
A federated learning server using FedAtt.

Reference:

Ji et al., "Learning Private Neural Language Modeling with Attentive Aggregation"
(https://arxiv.org/pdf/1812.07108.pdf)
"""
from collections import OrderedDict

from plato.servers import fedavg

import logging
import torch
import torch.nn.functional as F
import numpy as np


class Server(fedavg.Server):
    """A federated learning server using the FedAtt algorithm."""
    async def federated_averaging(self, reports):
        """Aggregate weight updates from the clients using FedAtt."""
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
                atts[name][i] = torch.linalg.norm(weight - delta)
            atts[name] = F.softmax(atts[name], dim=0)

        for name, weight in baseline_weights.items():
            att_weight = self.trainer.zeros(weight.shape)
            for i, update in enumerate(updates):
                delta = update[name]
                att_weight += (weight - delta).mul(atts[name][i])
            #TODO: plus random noise?
            # epsilon as an argument?
            att_update[name] = -att_weight.mul(1.2)

        # Load updated weights into model
        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + att_update[name]

        return updated_weights
