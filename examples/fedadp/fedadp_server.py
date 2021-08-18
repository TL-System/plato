"""
A federated learning training session using the FedAdp algorithm.

Reference:

Wu et al., "Fast-Convergent Federated Learning with Adaptive Weighting,"
in IEEE Transactions on Cognitive Communications and Networking (TCCN'21).

https://ieeexplore.ieee.org/abstract/document/9442814
"""

import math

import numpy as np

from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAdp algorithm."""
    def __init__(self):
        super().__init__()

        # alpha controls the decreasing rate of the mapping function
        self.alpha = 5
        self.local_correlations = {}
        self.last_global_grads = None
        self.adaptive_weighting = None

    def extract_client_updates(self, updates):
        """ Extract the model weights and update directions from clients updates. """
        weights_received = [payload for (__, payload) in updates]

        num_samples = [report.num_samples for (report, __) in updates]

        # Get adaptive weighting based on both node contribution and date size
        self.adaptive_weighting = self.calc_adaptive_weighting(weights_received, num_samples)

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        """ Aggregate weight updates and deltas updates from the clients. """
        # Extract weights udpates from the client updates
        weights_received = self.extract_client_updates(updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        # Use adaptive weighted average  
        for i, update in enumerate(weights_received):
            for name, delta in update.items():
                avg_update[name] += delta * self.adaptive_weighting[i]

        return avg_update

    def calc_adaptive_weighting(self, updates, num_samples):
        """ Compute the weights for model aggregation considering both node contribution
        and data size. """
        # Get the node contribution
        contribs = self.calc_contribution(updates)

        # Calculate the weighting of each participating client for aggregation
        adaptive_weighting = [None] * len(updates)
        total_weight = 0.0
        for i, contrib in enumerate(contribs):
            total_weight += num_samples[i] * math.exp(contrib)
        for i, contrib in enumerate(contribs):
            adaptive_weighting[i] = (num_samples[i] * math.exp(contrib)) / total_weight

        return adaptive_weighting

    def calc_contribution(self, updates):
        """ Calculate the node contribution based on the angle between the local
        and global gradients. """
        correlations, contribs = [None] * len(updates), [None] * len(updates)

        # Update the baseline model weights
        curr_global_grads = self.process_grad(self.algorithm.extract_weights())
        if self.last_global_grads is None:
            self.last_global_grads = np.zeros(len(curr_global_grads))
        global_grads = np.subtract(curr_global_grads, self.last_global_grads)
        self.last_global_grads = curr_global_grads

        # Compute angles in radian between local and global gradients
        for i, update in enumerate(updates):
            local_grads = self.process_grad(update)
            inner = np.inner(global_grads, local_grads)
            norms = np.linalg.norm(global_grads) * np.linalg.norm(local_grads)
            correlations[i] = np.arccos(np.clip(inner / norms, -1.0, 1.0))

        for i, correlation in enumerate(correlations):
            client_id = self.selected_clients[i]

            # Update the smoothed angle for all clients
            if client_id not in self.local_correlations.keys():
                self.local_correlations[client_id] = correlation
            self.local_correlations[client_id] = ((self.current_round - 1) 
            / self.current_round) * self.local_correlations[client_id] 
            + (1 / self.current_round) * correlation

            # Non-linear mapping to node contribution
            contribs[i] = self.alpha * (1 - math.exp(-math.exp(-self.alpha 
                          * (self.local_correlations[client_id] - 1))))

        return contribs

    @staticmethod
    def process_grad(grads):
        """Convert gradients to a flattened 1-D array."""
        grads = list(dict(sorted(grads.items(), key=lambda x: x[0].lower())).values())

        flattened = grads[0]
        for i in range(1, len(grads)):
            flattened = np.append(flattened, grads[i]) 

        return flattened
