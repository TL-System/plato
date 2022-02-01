import os
from dataclasses import dataclass
import statistics
from tokenize import Double
from plato.clients import simple
import torch
import numpy as np


@dataclass
class Report(simple.Report):
    """Client report sent to the FedSarah federated learning server."""
    payload_length: int
    #statistics_utility: float


class Client(simple.Client):
    """ A FedSarah federated learning client who sends weight updates
    and client control variates. """
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.statistical_utility = None

    async def train(self):
        """ Initialize the server control variates and client control variates for the trainer. """

        report, weights = await super().train()
        # compute statistical_utility
        filename = f"{self.client_id}__squred_batch_loss.pth"
        sum_loss = torch.load(filename).detach().numpy()

        self.statistical_utility = np.abs(report.num_samples) * np.sqrt(
            1.0 / report.num_samples * sum_loss)

        return Report(report.num_samples, report.accuracy,
                      report.training_time, report.update_response,
                      2), [weights, self.statistical_utility]
