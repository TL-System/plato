"""
A customized client for FedSarah.
"""
import os
from dataclasses import dataclass

from plato.clients import simple


@dataclass
class Report(simple.Report):
    """Client report sent to the FedSarah federated learning server."""
    payload_length: int


class Client(simple.Client):
    """A FedSarah federated learning client who sends weight updates
    and client control variates."""
    def __init__(self):
        super().__init__()
        self.client_control_variates = None
        self.server_control_variates = None
        self.new_client_control_variates = None
        self.fl_round_counter = 0

    async def train(self):
        # Initialize the server control variates and client control variates for the trainer
        if self.server_control_variates is not None:
            self.trainer.client_control_variates = self.client_control_variates
            self.trainer.server_control_variates = self.server_control_variates

        self.trainer.fl_round_counter = self.fl_round_counter
        self.fl_round_counter += 1

        report, weights = await super().train()

        # Get new client control variates from the trainer
        self.client_control_variates = self.trainer.new_client_control_variates

        fn = f"new_client_control_variates_{self.client_id}.pth"
        os.remove(fn)
        return Report(report.num_samples, report.accuracy,
                      report.training_time, report.data_loading_time,
                      2), [weights, self.client_control_variates]

    def load_payload(self, server_payload):
        "Load model weights and server control vairates from server payload onto this client"
        self.algorithm.load_weights(server_payload[0])

        #self.trainer.load_weights(server_payload[0])
        self.server_control_variates = server_payload[1]
