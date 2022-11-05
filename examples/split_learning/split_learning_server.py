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
from plato.utils import fonts
from plato.datasources import registry as datasources_registry


class Server(fedavg.Server):
    """The split learning server."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        # Split learning clients interact with server sequentially
        assert Config().clients.per_round == 1
        self.phase = "weights"
        self.clients_list = []
        self.client_last = None
        self.next_client = True
        self.gradients_to_send = {}
        self.test_accuracy = 0.0

    def choose_clients(self, clients_pool, clients_count):
        """Shuffle the clients and sequentially select them when the previous one is done."""
        if len(self.clients_list) == 0 and self.next_client:
            # Shuffle the client list
            self.clients_list = super().choose_clients(clients_pool, len(clients_pool))
            logging.warn(f"Client order: {self.clients_list}")

        if self.next_client:
            # Sequentially select clients
            self.client_last = [self.clients_list.pop(0)]
            self.next_client = False
        return self.client_last

    def customize_server_payload(self, payload):
        """Wrap up generating the server payload with any additional information."""
        if self.phase == "weights":
            # Split learning server doesn't send weights to client
            return (None, "weights")
        else:
            # Send gradients back to client to complete the training
            return (self.gradients_to_send.pop(self.selected_client_id), "gradients")

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        update = updates[0]
        report = update.report
        if report.type == "features":
            logging.warn("[%s] Features received, compute gradients.", self)
            feature_dataset = feature.DataSource([update.payload])

            # Training the model using all the features received from the client
            sampler = all_inclusive.Sampler(feature_dataset)
            self.algorithm.train(feature_dataset, sampler)

            # Compute the gradients and get ready to be sent
            self.gradients_to_send[update.client_id] = self._load_gradients()
            self.phase = "gradient"

        elif report.type == "weights":
            logging.warn("[%s] Weights received, start testing accuracy.", self)
            weights = update.payload

            # The weights after cut layer are not trained by clients
            self.algorithm.update_weights_before_cut(weights)

            # Manually Set up the testset since do_test is turned off in config
            if self.datasource is None:
                self.datasource = datasources_registry.get(client_id=0)
                self.testset = self.datasource.get_test_set()
                self.testset_sampler = all_inclusive.Sampler(
                    self.datasource, testing=True
                )
            self.test_accuracy = self.trainer.test(self.testset, self.testset_sampler)
            logging.warn(
                fonts.colourize(
                    f"[{self}] Global model accuracy: {100 * self.test_accuracy:.2f}%\n"
                )
            )
            self.phase = "weights"
            # Change client in next round
            self.next_client = True

        updated_weights = self.algorithm.extract_weights()
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

    def get_logged_items(self):
        """Overwrite the logged accuracy by latest test accuracy."""
        logged_items = super().get_logged_items()
        logged_items["accuracy"] = self.test_accuracy
        return logged_items
