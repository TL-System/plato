"""
A federated learning server using split learning.

Reference:

Vepakomma, et al., "Split Learning for Health: Distributed Deep Learning without Sharing
Raw Patient Data," in Proc. AI for Social Good Workshop, affiliated with ICLR 2018.

https://arxiv.org/pdf/1812.00564.pdf

Chopra, Ayush, et al. "AdaSplit: Adaptive Trade-offs for Resource-constrained Distributed
Deep Learning." arXiv preprint arXiv:2112.01637 (2021).

https://arxiv.org/pdf/2112.01637.pdf
"""

import logging

from plato.config import Config
from plato.datasources import feature
from plato.samplers import all_inclusive
from plato.servers import fedavg
from plato.utils import fonts
from plato.datasources import registry as datasources_registry


# pylint:disable=too-many-instance-attributes
class Server(fedavg.Server):
    """The split learning server."""

    # pylint:disable=too-many-arguments
    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        # Split learning clients interact with server sequentially
        assert Config().clients.per_round == 1
        self.phase = "prompt"
        self.clients_list = []
        self.client_last = None
        self.next_client = True
        self.test_accuracy = 0.0

    def choose_clients(self, clients_pool, clients_count):
        """Shuffle the clients and sequentially select them when the previous one is done."""
        if len(self.clients_list) == 0 and self.next_client:
            # Shuffle the client list
            self.clients_list = super().choose_clients(clients_pool, len(clients_pool))
            logging.warning("Client order: %s", str(self.clients_list))

        if self.next_client:
            # Sequentially select clients
            self.client_last = [self.clients_list.pop(0)]
            self.next_client = False
        return self.client_last

    def customize_server_payload(self, payload):
        """Wrap up generating the server payload with any additional information."""
        if self.phase == "prompt":
            # Split learning server doesn't send weights to client
            return (None, "prompt")
        return (self.trainer.get_gradients(), "gradients")

    # pylint: disable=unused-argument
    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate weight updates from the clients or train the model."""
        update = updates[0]
        report = update.report
        if report.type == "features":
            logging.warning("[%s] Features received, compute gradients.", self)
            feature_dataset = feature.DataSource([update.payload])

            # Training the model using all the features received from the client
            sampler = all_inclusive.Sampler(feature_dataset)
            self.algorithm.train(feature_dataset, sampler)

            self.phase = "gradient"
        elif report.type == "weights":
            logging.warning("[%s] Weights received, start testing accuracy.", self)
            weights = update.payload

            # The weights after cut layer are not trained by clients
            self.algorithm.update_weights_before_cut(weights)

            # Manually set up the testset since do_test is turned off in config
            if self.datasource is None:
                self.datasource = datasources_registry.get(client_id=0)
                self.testset = self.datasource.get_test_set()
                self.testset_sampler = all_inclusive.Sampler(
                    self.datasource, testing=True
                )

            self.test_accuracy = self.trainer.test(self.testset, self.testset_sampler)

            logging.warning(
                fonts.colourize(
                    f"[{self}] Global model accuracy: {100 * self.test_accuracy:.2f}%\n"
                )
            )
            self.phase = "prompt"
            # Change client in next round
            self.next_client = True

        updated_weights = self.algorithm.extract_weights()
        return updated_weights

    def clients_processed(self):
        # Replace the default accuracy by manually tested accuracy
        self.accuracy = self.test_accuracy
