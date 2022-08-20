"""
A federated learning client for FEI.
"""
import logging
import math
from types import SimpleNamespace

from plato.clients import simple
from plato.utils import fonts


class Client(simple.Client):
    """A federated learning client for FEI."""

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        loss = self.trainer.run_history.get_latest_metric("train_loss")
        logging.info(fonts.colourize(f"[Client #{self.client_id}] Loss value: {loss}"))
        report.valuation = self.calc_valuation(report.num_samples, loss)
        return report

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        # Resets the workload capacity at the initial step (for the new episode)
        if server_response["current_round"] == 1:
            # Resets the dataset
            self.datasource = None
            self.load_data()

    def calc_valuation(self, num_samples, loss):
        """Calculate the valuation value based on the number of samples and loss value."""
        valuation = float(1 / math.sqrt(num_samples)) * loss
        return valuation
