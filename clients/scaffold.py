"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (https://arxiv.org/pdf/1910.06378.pdf)
"""

import logging
import random

from dataclasses import dataclass
from config import Config
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

    def configure(self):
        """Prepare this client for training."""
        model_type = Config().trainer.model
        self.model = models_registry.get(model_type)
        self.trainer = trainers_registry.get(self.model, self.client_id)
        self.trainer.c_client = None
        self.trainer.c_server = None

    async def train(self):
        
        # update trainer.c_client & c_server
        """ """
        report = await super().train()

        # update clients' c_client and c_server 
        if self.c_client == None:
            self.c_client = {
                name: self.trainer.zeros(weights.shape)
                for name, weights in report.weights.items()
            }
        if self.c_server == None:
            self.c_server = {
                name: self.trainer.zeros(weights.shape)
                for name, weights in report.weights.items()
            }
        # update client.c_plus
        self.c_plus = self.trainer.c_plus

        # compute delta_c
        delta_c = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in report.weights.items()
        }

        delta_c = [self.c_plus - self.c_client in zip (self.c_plus, self.c_client]  # delta_c is a list

        self.c_client = self.c_plus

        return Report(report.client_id, report.num_samples, report.weights,
                      report.accuracy, report.training_time,
                      report.data_loading_time, delta_c)

    def process_server_response(self, server_response):
        """ Update c_server from the server response."""
        if 'c_server' in server_response:
            self.c_server = server_response['c_server']