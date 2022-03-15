"""
A federated learning client for FEI.
"""
import logging
import math
from dataclasses import dataclass
from plato.clients import simple
from plato.config import Config


@dataclass
class Report(simple.Report):
    """A client report to be sent to the FEI federated learning server."""
    client_id: int
    valuation: float


class Client(simple.Client):
    """ A federated learning client for FEI. """
    async def train(self):
        """Information of training loss will be reported after training the model."""

        logging.info("Training on FEI client #%d", self.client_id)

        report, weights = await super().train()

        # Get the valuation indicating the likely utility of training on this client
        loss = self.get_loss()
        valuation = self.calc_valuation(report.num_samples, loss)

        return Report(report.num_samples, report.accuracy,
                      report.training_time, report.comm_time,
                      report.update_response, self.client_id, valuation), weights

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        # Reset workload capacity at the initial step (for the new episode)
        if server_response['current_round'] == 1:
            # Reset dataset
            self.datasource = None
            self.load_data()

    def get_loss(self):
        """ Retrieve the loss value from the training process. """
        model_name = Config().trainer.model_name
        filename = f'{model_name}_{self.client_id}.loss'
        loss = self.trainer.load_loss(filename)
        return loss

    def calc_valuation(self, num_samples, loss):
        """ Calculate the valuation value based on the number of samples and loss value. """
        valuation = float(1 / math.sqrt(num_samples)) * loss
        return valuation
