"""
A customized client for FedSarah.
"""
import torch
from dataclasses import dataclass
from clients import simple
import os


@dataclass
class Report(simple.Report):
    """Client report sent to the FedSarah federated learning server."""
    payload_length: int


class FedSarahClient(simple.SimpleClient):
    """A FedSarah federated learning client who sends weight updates
    and client control variates."""
    def __init__(self):
        super().__init__()
        self.client_control_variates = None
        self.server_control_variates = None
        self.new_client_control_variates = None

    async def train(self):
        # Initialize the server control variates and client control variates for the trainer
        if self.server_control_variates is not None:
            self.trainer.client_control_variates = self.client_control_variates
            self.trainer.server_control_variates = self.server_control_variates

        report, weights = await super().train()

        # Get new client control variates from the trainer
        self.new_client_control_variates = self.trainer.new_client_control_variates

        # Compute deltas from client control variates
        deltas = []
        if self.client_control_variates is None:
            self.client_control_variates = [0] * len(
                self.new_client_control_variates)

        for client_control_variate, new_client_control_variate in zip(
                self.client_control_variates,
                self.new_client_control_variates):
            delta = torch.sub(new_client_control_variate,
                              client_control_variate)
            deltas.append(delta)

        # Update client control variates
        self.client_control_variates = self.new_client_control_variates
        fn = f"new_client_control_variates_{self.client_id}.pth"
        os.remove(fn)
        return Report(report.num_samples, report.accuracy,
                      report.training_time, report.data_loading_time,
                      2), [weights, deltas]

    def load_payload(self, server_payload):
        "Load model weights and server control vairates from server payload onto this client"
        self.trainer.load_weights(server_payload[0])
        self.server_control_variates = server_payload[1]
