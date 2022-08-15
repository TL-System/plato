"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import time
from types import SimpleNamespace

from plato.clients import edge


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""

    async def train(self):
        """The training process on a FedSaw edge client."""
        training_start_time = time.perf_counter()
        # Signal edge server to select clients to start a new round of local aggregation
        self.server.new_global_round_begins.set()

        # Wait for the edge server to finish model aggregation
        await self.server.model_aggregated.wait()
        self.server.model_aggregated.clear()

        # Extract model weights and biases
        weights = self.server.algorithm.extract_weights()

        average_accuracy = self.server.average_accuracy
        accuracy = self.server.accuracy

        training_time = time.perf_counter() - training_start_time

        comm_time = time.time()

        # Generate a report for the central server
        self._report = SimpleNamespace(
            num_samples=self.server.total_samples,
            accuracy=accuracy,
            training_time=training_time,
            comm_time=comm_time,
            update_response=False,
            average_accuracy=average_accuracy,
            client_id=self.client_id,
            comm_overhead=self.server.comm_overhead,
        )

        self.server.comm_overhead = 0

        return self.report, weights
