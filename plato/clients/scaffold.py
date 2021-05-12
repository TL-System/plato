"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" 
(https://arxiv.org/pdf/1910.06378.pdf)
"""

import os
from dataclasses import dataclass

import torch

from plato.clients import simple


@dataclass
class Report(simple.Report):
    """Client report sent to the SCAFFOLD federated learning server."""
    payload_length: int


class Client(simple.Client):
    """A SCAFFOLD federated learning client who sends weight updates
    and client control variate."""
    def __init__(self):
        super().__init__()
        self.client_update_direction = None
        self.server_update_direction = None
        self.new_client_update_direction = None

    async def train(self):
        # Initialize the server update direction and client update direction for trainer
        if self.server_update_direction is not None:
            self.trainer.client_update_direction = self.client_update_direction
            self.trainer.server_update_direction = self.server_update_direction

        report, weights = await super().train()

        # Get new client update direction from the trainer
        self.new_client_update_direction = self.trainer.new_client_update_direction

        # Compute deltas for update directions
        deltas = []
        if self.client_update_direction is None:
            self.client_update_direction = [0] * len(
                self.new_client_update_direction)

        for client_update_direction_, new_client_update_direction_ in zip(
                self.client_update_direction,
                self.new_client_update_direction):
            delta = torch.sub(new_client_update_direction_,
                              client_update_direction_)
            deltas.append(delta)

        # Update client update direction
        self.client_update_direction = self.new_client_update_direction
        fn = f"new_client_update_direction_{self.client_id}.pth"
        os.remove(fn)
        return Report(report.num_samples, report.accuracy,
                      report.training_time, report.data_loading_time,
                      2), [weights, deltas]

    def load_payload(self, server_payload):
        "Load model weights and server update direction from server payload onto this client"
        self.algorithm.load_weights(server_payload[0])
        self.server_update_direction = server_payload[1]
