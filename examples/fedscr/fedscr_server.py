"""
A federated learning server using FedSCR. The server extracts the model updates from each client,
aggregates them and adds them to the global model from the previous round.
"""

import asyncio
import os
import math

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """A federated learning server using the FedSCR algorithm."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # Loss variances for each communication round used by the adaptive algorithm
        self.variances = []

        # Local loss received from each client
        self.local_loss = []

        # Model divergences received from each client
        self.divs = {}

        # Average weight received updates from each client
        self.avg_update = {}

        self.mean_variance = None

        self.update_thresholds = {
            str(client_id): Config().clients.update_threshold
            if hasattr(Config().clients, "update_threshold")
            else 0.3
            for client_id in range(1, self.total_clients + 1)
        }

        self.orig_threshold = (
            Config().clients.update_threshold
            if hasattr(Config().clients, "update_threshold")
            else 0.3
        )

        # Hyperparameters used for the adaptive algorithm
        self.delta1 = (
            Config().clients.delta1 if hasattr(Config().clients, "delta1") else None
        )
        self.delta2 = (
            Config().clients.delta2 if hasattr(Config().clients, "delta2") else None
        )
        self.delta3 = (
            Config().clients.delta3 if hasattr(Config().clients, "delta3") else None
        )

    def customize_server_response(self, server_response: dict) -> dict:
        """Wrap up generating the server response with any additional information."""
        if self.trainer.use_adaptive and self.current_round > 1:
            self.calc_threshold()
            server_response["update_thresholds"] = self.update_thresholds
        return server_response

    def calc_threshold(self):
        """Calculate new update thresholds for each client."""
        for client_id in self.divs:
            sigmoid = (
                self.delta1 * self.divs[client_id]
                + self.delta2 * self.avg_update[client_id]
                + self.delta3 * self.mean_variance
            )
            self.update_thresholds[str(client_id)] = (
                1 / (1 + (math.e**-sigmoid))
            ) * self.orig_threshold

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        deltas_received = [update.payload for update in updates]

        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            num_samples = updates[i].report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    def weights_aggregated(self, updates):
        """Extract information from client reports. Called after weights have been aggregated"""

        if self.trainer.use_adaptive:
            self.local_loss = [update.report.loss for update in updates]
            self.mean_variance = self.calc_loss_var()
            self.divs = {
                update.client_id: update.report.div_from_global for update in updates
            }
            self.avg_update = {
                update.client_id: update.report.avg_update for update in updates
            }

    def calc_loss_var(self):
        """Calculate the loss variance using mean squared error."""
        global_loss = sum(self.local_loss) / len(self.local_loss)

        # Compute the mean squared error loss
        error = 0
        for loss in self.local_loss:
            error += math.pow((loss - global_loss), 2)
        mse_loss = 1 / len(self.local_loss) * error

        self.variances.append(mse_loss)

        mew = sum(self.variances)
        if self.current_round > 3:
            mew = mew * (1 / (self.current_round - 2))
        else:
            mew = 0
        return mew

    def server_will_close(self):
        """Method called at the start of closing the server."""
        model_name = Config().trainer.model_name
        checkpoint_path = Config().params["checkpoint_path"]

        # Delete files created by the clients.
        for client_id in range(1, self.total_clients + 1):
            acc_grad_file = f"{checkpoint_path}/{model_name}_client{client_id}_grad.pth"
            os.remove(acc_grad_file)
