"""
A federated learning training session using the FedAdp algorithm.

Reference:

Wu et al., "Fast-Convergent Federated Learning with Adaptive Weighting,"
in IEEE Transactions on Cognitive Communications and Networking (TCCN'21).

https://ieeexplore.ieee.org/abstract/document/9442814
"""

import math

import numpy as np
from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAdp algorithm."""
    def __init__(self):
        super().__init__()

        self.local_angles = {}
        self.last_global_grads = None
        self.adaptive_weighting = None
        self.global_grads = None

    async def federated_averaging(self, updates):
        """ Aggregate weight updates and deltas updates from the clients. """
        # Extract weights udpates from the client updates
        weights_received = self.extract_client_updates(updates)

        num_samples = [report.num_samples for (report, __, __) in updates]
        total_samples = sum(num_samples)

        self.global_grads = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            for name, delta in update.items():
                self.global_grads[name] += delta * (num_samples[i] /
                                                    total_samples)

        # Get adaptive weighting based on both node contribution and date size
        self.adaptive_weighting = self.calc_adaptive_weighting(
            weights_received, num_samples)

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
            adaptive_weighting[i] = (num_samples[i] *
                                     math.exp(contrib)) / total_weight

        return adaptive_weighting

    def calc_contribution(self, updates):
        """ Calculate the node contribution based on the angle between the local
        and global gradients. """
        angles, contribs = [None] * len(updates), [None] * len(updates)

        # Compute the global gradient which is surrogated by using local gradients
        self.global_grads = self.process_grad(self.global_grads)

        # Compute angles in radian between local and global gradients
        for i, update in enumerate(updates):
            local_grads = self.process_grad(update)
            inner = np.inner(self.global_grads, local_grads)
            norms = np.linalg.norm(
                self.global_grads) * np.linalg.norm(local_grads)
            angles[i] = np.arccos(np.clip(inner / norms, -1.0, 1.0))

        for i, angle in enumerate(angles):
            client_id = self.selected_clients[i]

            # Update the smoothed angle for all clients
            if client_id not in self.local_angles.keys():
                self.local_angles[client_id] = angle
            self.local_angles[client_id] = (
                (self.current_round - 1) / self.current_round
            ) * self.local_angles[client_id] + (1 / self.current_round) * angle

            # Non-linear mapping to node contribution
            alpha = Config().algorithm.alpha if hasattr(
                Config().algorithm, 'alpha') else 5

            contribs[i] = alpha * (
                1 - math.exp(-math.exp(-alpha *
                                       (self.local_angles[client_id] - 1))))

        return contribs

    @staticmethod
    def process_grad(grads):
        """Convert gradients to a flattened 1-D array."""
        grads = list(
            dict(sorted(grads.items(), key=lambda x: x[0].lower())).values())

        flattened = grads[0]
        for i in range(1, len(grads)):
            flattened = np.append(flattened,
                                  -grads[i] / Config().trainer.learning_rate)

        return flattened
