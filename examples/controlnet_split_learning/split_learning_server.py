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
# pylint:disable=import-error
import logging
import time
import torch


from split_learning import split_learning_server
from plato.datasources import feature
from plato.samplers import all_inclusive
from plato.utils import fonts
from plato.config import Config


# pylint:disable=attribute-defined-outside-init
# pylint:disable=too-few-public-methods
# pylint:disable=too-many-arguments
# pylint:disable=too-many-instance-attributes
class Server(split_learning_server.Server):
    """The split learning server."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.server_time = 0
        self.client_time = 0
        self.client_mem = 0
        self.server_mem = 0

    # pylint: disable=unused-argument
    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate weight updates from the clients or train the model."""
        update = updates[0]
        report = update.report
        self.client_time = report.training_time
        if "cuda" in Config().device():
            self.client_mem = report.gpu_mem
        if report.type == "features":
            logging.warning("[%s] Features received, compute gradients.", self)
            feature_dataset = feature.DataSource([update.payload])

            # Training the model using all the features received from the client
            sampler = all_inclusive.Sampler(feature_dataset)
            torch.cuda.synchronize()
            tic = time.perf_counter()
            self.algorithm.train(feature_dataset, sampler)
            torch.cuda.synchronize()
            toc = time.perf_counter()
            self.server_time = toc - tic
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_max_memory_allocated()
            self.server_mem = gpu_mem

            self.phase = "gradient"
        elif report.type == "weights":
            logging.warning("[%s] Weights received, start testing accuracy.", self)
            weights = update.payload

            # The weights after cut layer are not trained by clients
            self.algorithm.update_weights_before_cut(weights)

            # Manually set up the testset since do_test is turned off in config
            self.testset = self.custom_datasource().get_test_set()
            self.testset_sampler = all_inclusive.Sampler(
                self.custom_datasource(), testing=True
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

    def get_logged_items(self) -> dict:
        """Log more information including computation time and the flops and memory."""
        logged_items = super().get_logged_items()
        logged_items["server_time"] = self.server_time
        logged_items["client_time"] = self.client_time
        logged_items["server_mem"] = self.server_mem
        logged_items["client_mem"] = self.client_mem
        return logged_items
