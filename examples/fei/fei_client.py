"""
A federated learning client for FEI.
"""
import math
from types import SimpleNamespace

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A federated learning client for FEI."""

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        loss = self.get_loss()
        report.valuation = self.calc_valuation(report.num_samples, loss)
        return report

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        # Reset workload capacity at the initial step (for the new episode)
        if server_response["current_round"] == 1:
            # Reset dataset
            self.datasource = None
            self.load_data()

    def get_loss(self):
        """Retrieve the loss value from the training process."""
        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}.loss"
        loss = self.trainer.load_loss(filename)
        return loss

    def calc_valuation(self, num_samples, loss):
        """Calculate the valuation value based on the number of samples and loss value."""
        valuation = float(1 / math.sqrt(num_samples)) * loss
        return valuation
