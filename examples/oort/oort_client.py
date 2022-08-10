"""
A federated learning client that sends its statistical utility
"""

from dataclasses import dataclass
import numpy as np
import torch

from plato.clients import simple
from plato.config import Config


@dataclass
class Report(simple.Report):
    """Client report sent to the federated learning server."""

    statistics_utility: float


class Client(simple.Client):
    """
    A federated learning client that calculates its statistical utility
    """

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.statistical_utility = None

    def customize_report(self, report):
        """Wrap up generating the report with any additional information."""
        model_name = Config().trainer.model_name
        model_path = Config().params["checkpoint_path"]
        filename = f"{model_path}/{model_name}_{self.client_id}_squared_batch_loss.pth"
        sum_loss = torch.load(filename)
        self.statistical_utility = np.abs(report.num_samples) * np.sqrt(
            1.0 / report.num_samples * sum_loss
        )
        setattr(report, "statistics_utility", self.statistical_utility)
        return report
