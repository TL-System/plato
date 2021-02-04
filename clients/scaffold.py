"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" 
(https://arxiv.org/pdf/1910.06378.pdf)
"""

import torch
from dataclasses import dataclass
from clients import simple


@dataclass
class Report(simple.Report):
    """A client report containing c."""
    delta_c: list


class ScaffoldClient(simple.SimpleClient):
    """A Scaffold federated learning client who sends weight updates
    and c."""
    def __init__(self):
        super().__init__()
        self.c_client = None
        self.c_server = None
        self.c_plus = None

    async def train(self):

        # set up trainer.c_client & trainer.c_server
        if self.c_server is not None:
            self.trainer.c_client = self.c_client
            self.trainer.c_server = self.c_server

        report = await super().train()

        # update client.c_plus from local trainer
        self.c_plus = self.trainer.c_plus

        # compute delta_c
        delta_c = []
        if self.c_client is None:
            self.c_client = [0] * len(self.c_plus)

        for c_client_, c_plus_ in zip(self.c_client, self.c_plus):
            delta = torch.sub(c_plus_, c_client_)
            delta_c.append(delta)

        # update c_client by c_plus
        self.c_client = self.c_plus

        return Report(report.client_id, report.num_samples, report.weights,
                      report.accuracy, report.training_time,
                      report.data_loading_time, delta_c)

    def load_c_server(self, c_server):
        """Load c_server from server_response."""
        self.c_server = c_server
