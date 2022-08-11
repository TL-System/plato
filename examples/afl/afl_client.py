"""
A federated learning server using Active Federated Learning, where in each round
clients are selected not uniformly at random, but with a probability conditioned
on the current model, as well as the data on the client, to maximize efficiency.

Reference:

Goetz et al., "Active Federated Learning", 2019.

https://arxiv.org/pdf/1909.12641.pdf
"""
import math
from types import SimpleNamespace

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A federated learning client for AFL."""

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        loss = self.get_loss()
        report.valuation = self.calc_valuation(report.num_samples, loss)
        return report

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
