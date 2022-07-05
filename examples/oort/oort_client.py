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

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.statistical_utility = None

    async def train(self):
        """ Regular training process on a FL client. """
        report, weights = await super().train()

        model_name = Config().trainer.model_name
        model_path = Config().params['checkpoint_path']
        filename = f"{model_path}/{model_name}_{self.client_id}__squred_batch_loss.pth"
        sum_loss = torch.load(filename).detach().cpu().numpy()
        self.statistical_utility = np.abs(report.num_samples) * np.sqrt(
            1.0 / report.num_samples * sum_loss)

        return Report(report.num_samples, report.accuracy,
                      report.training_time, report.comm_time,
                      report.update_response,
                      self.statistical_utility), weights
