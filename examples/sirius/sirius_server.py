"""
An asynchronous federated learning server using Sirius.
"""

import torch
import asyncio
from plato.config import Config
from plato.servers import fedavg

class Server(fedavg.Server):
    """A federated learning server using the sirius algorithm."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.staleness_factor = 0.5

    def configure(self):
        """Initialize necessary variables."""
        super().configure()

        self.client_utilities = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging with calcuated staleness factor."""
        
        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(delta.shape)
            for name, delta in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            report = updates[i].report
            num_samples = report.num_samples
            staleness = updates[i].staleness
            staleness_factor = self.staleness_function(staleness)

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples) * staleness_factor # no normalization but from Zhifeng's code.

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    def staleness_function(self, stalenss):
        return 1.0 / pow(stalenss + 1, self.staleness_factor) # formula obtained from Zhifeng's code. (clients_manager/sirius/staleness_factor_calculator)

    def weights_aggregated(self, updates):
        """Method called at the end of aggregating received weights."""
        """Calculate client utility here and update the record on the server"""
        for update in updates:
            self.client_utilities[update.client_id] = update.report.statistics_utility * self.staleness_function(update.staleness)
            
           
