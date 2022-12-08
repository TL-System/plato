"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import time
from types import SimpleNamespace

from plato.clients import simple
from plato.config import Config
from plato.processors import registry as processor_registry


class Client(simple.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""

    def __init__(
        self,
        server,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.server = server

    def configure(self) -> None:
        """Prepare this edge client for training."""
        super().configure()

        # Pass inbound and outbound data payloads through processors for
        # additional data processing
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Client", client_id=self.client_id, trainer=self.server.trainer
        )

    def load_data(self) -> None:
        """The edge client does not need to train models using local data."""

    def _load_payload(self, server_payload) -> None:
        """The edge client loads the model from the central server."""
        self.server.algorithm.load_weights(server_payload)

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if "current_global_round" in server_response:
            self.server.current_global_round = server_response["current_global_round"]

    async def _train(self):
        """The aggregation workload on an edge client."""
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

        if (
            hasattr(Config().clients, "sleep_simulation")
            and Config().clients.sleep_simulation
        ):
            training_time = self.server.edge_training_time
            self.server.edge_training_time = 0
        else:
            training_time = time.perf_counter() - training_start_time

        comm_time = time.time()

        edge_server_comm_time = self.server.edge_comm_time
        self.server.edge_comm_time = 0

        # Generate a report for the central server
        report = SimpleNamespace(
            client_id=self.client_id,
            num_samples=self.server.total_samples,
            accuracy=accuracy,
            training_time=training_time,
            comm_time=comm_time,
            update_response=False,
            average_accuracy=average_accuracy,
            edge_server_comm_overhead=self.server.comm_overhead,
            edge_server_comm_time=edge_server_comm_time,
        )

        self._report = self.customize_report(report)

        self.server.comm_overhead = 0

        return self._report, weights
