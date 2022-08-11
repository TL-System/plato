"""
A federated learning client that sends its statistical utility
"""

import numpy as np
import torch
from types import SimpleNamespace

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """
    A federated learning client that calculates its statistical utility
    """

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.statistical_utility = None

    def customize_report(self, report) -> SimpleNamespace:
        """Wrap up generating the report with any additional information."""
        model_name = Config().trainer.model_name
        model_path = Config().params["checkpoint_path"]
        filename = f"{model_path}/{model_name}_{self.client_id}_squared_batch_loss.pth"
        sum_loss = torch.load(filename)
        self.statistical_utility = np.abs(report.num_samples) * np.sqrt(
            1.0 / report.num_samples * sum_loss
        )
        report.statistics_utility = self.statistical_utility
        return report
