"""
A federated learning client at the edge server in a cross-silo training workload.
"""

from dataclasses import dataclass
import time

from plato.clients import base
from plato.clients import simple
from plato.processors import registry as processor_registry


@dataclass
class Report(simple.Report):
    """ Client report, to be sent to the federated learning server. """
    average_accuracy: float
    client_id: str


class Client(base.Client):
    """ A federated learning client at the edge server in a cross-silo training workload. """
    def __init__(self, server):
        super().__init__()
        self.server = server
        self.report = None

    def configure(self):
        """ Prepare this edge client for training. """
        super().configure()

        # Pass inbound and outbound data payloads through processors for
        # additional data processing
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Client", client_id=self.client_id, trainer=self.server.trainer)

    def load_data(self):
        """ The edge client does not need to train models using local data. """

    def load_payload(self, server_payload) -> None:
        """ The edge client loads the model from the central server. """
        self.server.algorithm.load_weights(server_payload)

    def process_server_response(self, server_response):
        """ Additional client-specific processing on the server response. """
        if 'current_global_round' in server_response:
            self.server.current_global_round = server_response[
                'current_global_round']

    async def train(self):
        """ The aggregation workload on an edge client. """
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
                             self.client_id)

        return self.report, weights

    async def obtain_model_update(self, wall_time):
        """ Retrieving a model update corresponding to a particular wall clock time. """
        model = self.server.trainer.obtain_model_update(wall_time)
        weights = self.server.algorithm.extract_weights(model)
        self.report.update_response = True

        return self.report, weights

    def save_model(self, model_checkpoint):
        """ Saving the current aggregated model to a model checkpoint. """
        self.server.trainer.save_model(model_checkpoint)

    def load_model(self, model_checkpoint):
        """ Loading the current aggregated model from a model checkpoint. """
        self.server.trainer.load_model(model_checkpoint)
