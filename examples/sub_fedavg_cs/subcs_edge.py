"""
A federated learning client at the edge server in a cross-silo training workload.
"""

from dataclasses import dataclass
import time

from plato.clients import edge


@dataclass
class Report(edge.Report):
    """ Client report, to be sent to the federated learning server. """
    comm_overhead: float


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""
    async def train(self):
        """ The training process on a FedSaw edge client. """
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
        self.report = Report(self.server.total_samples, accuracy,
                             training_time, comm_time, False, average_accuracy,
                             self.client_id, self.server.comm_overhead)

        self.server.comm_overhead = 0

        return self.report, weights
