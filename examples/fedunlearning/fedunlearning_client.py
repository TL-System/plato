"""
A customized client for federated unlearning.

Federated unlearning allows clients to proactively erase their data from a trained model. The model
will be retrained from scratch during the unlearning process.

If the AdaHessian optimizer is used, it will reflect what the following paper proposed:

Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid
Retraining," in Proc. INFOCOM, 2022.

Reference: https://arxiv.org/abs/2203.07320
"""
import logging

import unlearning_iid

from plato.config import Config
from plato.utils.lib_mia import mia_client


class Client(mia_client.Client):
    """A federated learning client of federated unlearning."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=None,
        )

        self.previous_round = {}
        self.unlearning_clients = []

    def process_server_response(self, server_response):
        """
        Register the client when the retraining happens (communication round rollback).
        """
        if self.client_id in self.previous_round:
            previous_round = self.previous_round[self.client_id]
        else:
            previous_round = 0

        client_pool = Config().clients.clients_requesting_deletion

        if self.client_id in client_pool and self.current_round <= previous_round:
            if self.client_id not in self.unlearning_clients:
                self.unlearning_clients.append(self.client_id)

        self.previous_round[self.client_id] = self.current_round

    def configure(self):
        """
        If a client requested deletion, replace its sampler accordingly in the
        retraining phase.
        """
        super().configure()

        if self.client_id in self.unlearning_clients:
            logging.info(
                "[%s] Unlearning sampler deployed: %s%% of the samples were deleted.",
                self,
                Config().clients.deleted_data_ratio * 100,
            )

            self.sampler = unlearning_iid.Sampler(
                self.datasource, self.client_id, testing=False
            )
