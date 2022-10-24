"""
A federated learning server using split learning.

Reference:

Vepakomma, et al., "Split learning for health: Distributed deep learning without sharing
raw patient data," in Proc. AI for Social Good Workshop, affiliated with ICLR 2018.

https://arxiv.org/pdf/1812.00564.pdf
"""

import logging
import os

import torch
from plato.config import Config
from plato.datasources import feature
from plato.samplers import all_inclusive
from plato.servers import fedavg


class Server(fedavg.Server):
    """The split learning server."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)

        self.phase = "weights"
        self.clients_selected_last = None
        self.gradients_to_send = {}

    def choose_clients(self, clients_pool, clients_count):
        """Choose the same clients every two rounds."""
        if self.phase == "weights":
            self.clients_selected_last = super().choose_clients(
                clients_pool, clients_count
            )

        return self.clients_selected_last

    def customize_server_payload(self, payload):
        """Wrap up generating the server payload with any additional information."""
        if self.phase == "weights":
            # Send global model weights to the clients
            return (payload, "weights")
        else:
            # Send gradients back to client to complete the training
            return (self.gradients_to_send.pop(self.selected_client_id), "gradients")

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Compute gradients and aggregate weights in different rounds."""
        if self.phase == "weights":
            for update in updates:
                feature_dataset = feature.DataSource([update.payload])

                # Training the model using all the features received from the client
                sampler = all_inclusive.Sampler(feature_dataset)
                self.algorithm.train(feature_dataset, sampler)

                # Compute the gradients and get ready to be sent
                self.gradients_to_send[update.client_id] = self._load_gradients()

            self.phase = "gradients"
            # No weights update in this round
            return baseline_weights

        elif self.phase == "gradients":
            # Perform federated averaging algorithm (copied from fedavg.Server for convenience)
            self.total_samples = sum(update.report.num_samples for update in updates)
            updated_weights = {
                name: self.trainer.zeros(weight.shape)
                for name, weight in weights_received[0].items()
            }

            for i, update in enumerate(weights_received):
                report = updates[i].report
                num_samples = report.num_samples

                for name, weight in update.items():
                    # Use weighted average by the number of samples
                    updated_weights[name] += weight * (num_samples / self.total_samples)

            self.phase = "weights"
            return updated_weights

    def _load_gradients(self):
        """Loading gradients from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        model_gradients_path = f"{model_path}/{model_name}_gradients.pth"
        logging.info(
            "[Server #%d] Loading gradients from %s.", os.getpid(), model_gradients_path
        )

        return torch.load(model_gradients_path)
