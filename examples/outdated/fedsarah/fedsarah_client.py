"""
A customized client for FedSarah.

Reference: Ngunyen et al., "SARAH: A Novel Method for Machine Learning Problems
Using Stochastic Recursive Gradient." (https://arxiv.org/pdf/1703.00102.pdf)

"""
import os
import time
from types import SimpleNamespace

from plato.clients import simple


class Client(simple.Client):
    """A FedSarah federated learning client who sends weight updates
    and client control variates."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.client_control_variates = None
        self.server_control_variates = None
        self.new_client_control_variates = None
        self.fl_round_counter = 0

    async def _train(self):
        """Initialize the server control variates and client control variates for the trainer."""
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

        comm_time = time.time()
        return SimpleNamespace(
            num_samples=report.num_samples,
            accuracy=report.accuracy,
            training_time=report.training_time,
            comm_time=comm_time,
            update_response=report.update_response,
            payload_length=2,
        ), [weights, self.client_control_variates]

    def _load_payload(self, server_payload):
        """Load model weights and server control vairates from server payload onto this client."""
        self.algorithm.load_weights(server_payload[0])
        self.server_control_variates = server_payload[1]
