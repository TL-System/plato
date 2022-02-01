"""
A federated learning training session using utility evaluation.
"""
from re import T
from tkinter.messagebox import NO
from plato.config import Config
from plato.servers import fedavg
import numpy as np
import asyncio


class Server(fedavg.Server):
    """A federated learning server using the Fedut algorithm."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)
        self.statistical_utility = None
        self.global_utility = None
        self.system_utility = None
        self.local_training_time = None
        #print(Config.server)
        self.expected_duration = Config().server.expected_duration

    def extract_client_updates(self, updates):
        """ Extract the model weights and statistical utility from clients updates. """
        weights_received = [payload[0] for (__, payload, __) in updates]

        self.statistical_utility = [
            payload[1] for (__, payload, __) in updates
        ]
        # Extract the local training time
        self.local_training_time = [
            report.training_time for (report, __, __) in updates
        ]

        return self.algorithm.compute_weight_updates(weights_received)

    async def federated_averaging(self, updates):
        """ Aggregate weight and delta updates from client updates. """
        weights_received = self.extract_client_updates(updates)

        # Adjust expected duration

        # Compute system utility by t_i
        self.system_utility = [
            pow(self.expected_duration, t) *
            np.maximum(np.sign(self.expected_duration - t), 0)
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

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __, __) in updates])

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            report, __, __ = updates[i]
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += self.global_utility[i] * delta * (
                    num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update