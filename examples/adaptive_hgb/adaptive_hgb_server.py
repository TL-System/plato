"""
The defined server for the Adaptive hierarchical gradient blending method

"""

import asyncio

from plato.servers import fedavg


class Server(fedavg.Server):
    """Federated server for adaptive gradient blending."""
    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        weights_received = self.extract_client_updates(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __) in updates])

        clients_delta_ogs = [(report.delta_O, report.delta_G)
                             for (report, __) in updates]

        clients_optimal_weights = self.get_optimal_gradient_blend_weights_OG(
            delta_OGs=clients_delta_ogs)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            report, __ = updates[i]

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (clients_optimal_weights[i])

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update
