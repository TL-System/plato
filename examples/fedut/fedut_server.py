"""
A federated learning training session using utility evaluation.
"""
from re import T
from tkinter.messagebox import NO
from plato.config import Config
from plato.servers import fedavg
import numpy as np


class Server(fedavg.Server):
    """A federated learning server using the Fedut algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.statistical_utility = None
        self.global_utility = None
        self.system_utility = None

    def extract_client_updates(self, updates):
        """ Extract the model weights and statistical utility from clients updates. """
        weights_received = [payload[0] for (__, payload, __) in updates]

        self.statistical_utility = [
            payload[1] for (__, payload, __) in updates
        ]

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        """ Aggregate weight and delta updates from client updates. """
        avg_update = await super().federated_averaging(updates)

        # Initialize global utility
        self.global_utility = [0] * len(self.statistical_utility[0])

        # Compute system utility by t_i
        self.system_utility = [
            pow(T, t) * np.maximum(np.sign(T - t), 0)
            for t in self.local_training_time
        ]
        # Compute global utility
        self.global_utility = [
            stats * sys for stats, sys in zip(self.statistical_utility,
                                              self.system_utility)
        ]
        # Normalize global utility via softmax
        self.global_utility = [
            np.exp(U) / np.sum(np.exp(self.global_utility))
            for U in self.global_utility
        ]

        # Update server model
        agg_update = []
        for update, uti in zip(avg_update, self.global_utility):
            agg_update.append(update * uti)

        return agg_update