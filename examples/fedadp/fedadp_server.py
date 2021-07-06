"""
A federated learning training session using FedAdp.

Reference:

Wu et al., "Fast-Convergent Federated Learning with Adaptive Weighting,"
in IEEE Transactions on Cognitive Communications and Networking (TCCN'21).

https://ieeexplore.ieee.org/abstract/document/9442814
"""
from collections import OrderedDict

from plato.servers import fedavg

import logging
import numpy as np
import math
import random


class Server(fedavg.Server):
    """A federated learning server using the FedAdp algorithm."""
    def __init__(self):
        super().__init__()
        # alpha controls the decreasing rate of the mapping function
        self.alpha = 5
        self.local_correlations = {}
        self.last_global_weights = None

    async def federated_averaging(self, updates):
        """ Aggregate weight updates and deltas updates from the clients. """
        # Extract weights from the updates
        weights_received = self.extract_client_updates(updates)

        # Get the number of samples
        num_samples = [report.num_samples for (report, __) in updates]

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        # Use weighted average based on both node contribution and date size
        adaptive_weighting = self.adaptive_weighting(weights_received, num_samples)
        for i, update in enumerate(weights_received):
            for name, delta in update.items():
                avg_update[name] += delta * adaptive_weighting[i]

        return avg_update

    def adaptive_weighting(self, updates, num_samples):
        """Compute the weights for model aggregation considering both node contribution and data size.""" 
        contribs = self.calc_contribution(updates)
        
        # Calculate the weighting of participating clients
        adaptive_weighting = [None] * len(updates)
        total_weight = 0.0
        for i, contrib in enumerate(contribs):
            total_weight += num_samples[i] * math.exp(contrib)
        for i, contrib in enumerate(contribs):
            adaptive_weighting[i] = (num_samples[i] * math.exp(contrib)) / total_weight

        return adaptive_weighting
    
    def calc_contribution(self, updates):
        correlations, contribs = [None] * len(updates), [None] * len(updates)
        # Update the baseline model weights
        curr_global_weights = self.algorithm.extract_weights()
        # Convert to list as vector
        curr_global_weights = list(dict(sorted(curr_global_weights.items(), key=lambda x: x[0].lower())).values())
        if not self.last_global_weights:
            self.last_global_weights = curr_global_weights
        global_gradients = [curr - last for curr, last in zip(curr_global_weights, self.last_global_weights)]
        self.last_global_weights = curr_global_weights

        # Compute angles in radian between local and global gradients
        for i, update in enumerate(updates):
            local_gradients = list(dict(sorted(update.items(), key=lambda x: x[0].lower())).values())
            inner = np.inner(global_gradients, local_gradients)
            norms = np.linalg.norm(global_gradients) * np.linalg.norm(local_gradients)
            cos = inner / norms
            correlation[i] = np.arccos(np.clip(cos), -1.0, 1.0)
        
        for i, correlation in enumerate(correlations):
            client_id = self.select_clients[i]
            # Update the smoothed angle for all clients
            self.local_correlations[client_id] = ((self.current_round - 1) / self.current_round) * self.local_correlations[client_id] 
            + (1 / self.current_round) * correlation
            # Non-linear mapping to node contribution
            contribs[i] = self.alpha * (1 - math.exp(-math.exp(-self.alpha * (self.local_correlations[client_id] - 1))))
            
        return contribs
