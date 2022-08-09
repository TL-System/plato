"""
A federated learning server using FedSCR. The server extracts the model updates from each client,
aggregates them and adds them to the global model from the previous round.
"""

import asyncio
import logging
import os
import math
import pickle
import torch

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """A federated learning server using the FedSCR algorithm."""

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # loss variances for each communication round used by the adaptive algorithm.
        self.variances = []

        # local loss received from each client
        self.local_loss = []

        # model divergences received from each client
        self.divs = {}

        # average weight received updates from each client
        self.avg_update = {}

        self.mean_variance = None

        self.update_threshold = [
            Config().clients.update_threshold
            if hasattr(Config().clients, "update_threshold")
            else 0.3
        ] * self.total_clients

        self.orig_threshold = (
            Config().clients.update_threshold
            if hasattr(Config().clients, "update_threshold")
            else 0.3
        )

        # Hyperparameters used for the adaptive algorithm.
        self.delta1 = (
            Config().clients.delta1 if hasattr(Config().clients, "delta1") else None
        )
        self.delta2 = (
            Config().clients.delta2 if hasattr(Config().clients, "delta2") else None
        )
        self.delta3 = (
            Config().clients.delta3 if hasattr(Config().clients, "delta3") else None
        )

    def configure(self):
        """Log the usage of either the adaptive or normal algorithm."""
        super().configure()
        if self.trainer.use_adaptive:
            logging.info("Using adaptive algorithm.")

    def extract_client_updates(self, updates):
        """Extract the model weight updates from clients."""

        deltas_received = [payload for (__, __, payload, __) in updates]
        self.local_loss = [report.loss for (__, report, __, __) in updates]
        if self.trainer.use_adaptive:
            self.divs = {
                client_id: report.div for (client_id, report, __, __) in updates
            }
            self.avg_update = {
                client_id: report.avg_update for (client_id, report, __, __) in updates
            }
        return deltas_received

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        deltas_received = self.extract_client_updates(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (__, report, __, __) in updates]
        )

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            __, report, __, __ = updates[i]
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    async def process_reports(self):
        """Process the client reports by aggregating their weights."""
        await self.aggregate_weights(self.updates)
        config = Config().trainer._asdict()

        # Testing the global model accuracy
        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                "[%s] Average client accuracy: %.2f%%.", self, 100 * self.accuracy
            )
        else:
            # Testing the updated model directly at the server
            self.accuracy = self.trainer.test_model(
                config, self.testset, self.testset_sampler
            )

        if hasattr(Config().trainer, "target_perplexity"):
            logging.info("[%s] Global model perplexity: %.2f\n", self, self.accuracy)
        else:
            logging.info(
                "[%s] Global model accuracy: %.2f%%\n", self, 100 * self.accuracy
            )

        if self.trainer.use_adaptive:
            self.mean_variance = self.calc_loss_var()

    def calc_loss_var(self):
        """Calculates the loss variance using mean squared error."""
        global_loss = [sum(self.local_loss) / len(self.local_loss)]
        loss = torch.nn.MSELoss()
        variance = loss(
            torch.FloatTensor(self.local_loss), torch.FloatTensor(global_loss)
        )
        self.variances.append(variance.data.item())

        mew = sum(variance for variance in self.variances)
        if self.current_round > 3:
            mew = mew * (1 / (self.current_round - 2))
        else:
            mew = 0
        return mew

    async def select_clients(self, for_next_batch=False):
        """Sends the update_thresholds to the clients."""
        await super().select_clients(for_next_batch)

        if self.trainer.use_adaptive:
            self.calc_threshold()
            model_path = Config().params["checkpoint_path"]
            model_name = Config().trainer.model_name

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            loss_path = f"{model_path}/{model_name}_thresholds.pkl"

            with open(loss_path, "wb") as file:
                pickle.dump(self.update_threshold, file)

    def calc_threshold(self):
        """Calculates new update thresholds for each client."""
        if self.current_round == 1:
            return
        for client_id in self.divs:
            sigmoid = (
                self.delta1 * self.divs[client_id]
                + self.delta2 * self.avg_update[client_id]
                + self.delta3 * self.mean_variance
            )
            self.update_threshold[client_id - 1] = (
                1 / (1 + (math.e**-sigmoid))
            ) * self.orig_threshold

    def server_will_close(self):
        """
        Method called at the start of closing the server.
        """
        # Delete gradient files created by the clients.
        model_name = Config().trainer.model_name
        checkpoint_path = Config().params["checkpoint_path"]
        for client_id in range(1, self.total_clients + 1):
            allgrad_path = f"{checkpoint_path}/{model_name}_client{client_id}_grad.pth"
            if os.path.exists(allgrad_path):
                os.remove(allgrad_path)
