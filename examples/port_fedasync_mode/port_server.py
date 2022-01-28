"""
A federated learning server using Port.

Reference:

"How Asynchronous can Federated Learning Be?"

"""

import asyncio
import logging
import os
from collections import OrderedDict

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAsync algorithm. """

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        weights_received = self.extract_client_updates(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __, __) in updates])

        clients_staleness = [staleness for (__, __, staleness) in updates]

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            report, __, __ = updates[i]
            num_samples = report.num_samples
            staleness_factor = Server.staleness_function(clients_staleness[i])

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * staleness_factor * (
                    num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    @staticmethod
    def staleness_function(staleness):

        # Add a hyperprameters to tune the staleness_function
        staleness_hyperpara = 0.5

        staleness_factor = staleness_hyperpara / (staleness +
                                                  staleness_hyperpara)
        print(
            "***********************enter in staleness function ****************************"
        )
        print("staleness is: ", staleness)
        print("staleness_factor is: ", staleness_factor)
        return staleness_factor
