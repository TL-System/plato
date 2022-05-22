"""
A customized client for federated unlearning.

Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid
Retraining," in Proc. INFOCOM, 2022.

Reference: https://arxiv.org/abs/2203.07320
"""
import logging

import unlearning_iid
from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A federated learning client of federated unlearning."""

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model=model,
                         datasource=datasource,
                         algorithm=algorithm,
                         trainer=trainer)

        self.previous_round = {}

    def process_server_response(self, server_response):
        """
        If a client requested deletion, replace its sampler accordingly in the
        retraining phase.
        """
        if self.client_id in self.previous_round:
            previous_round = self.previous_round[self.client_id]
        else:
            previous_round = 0

        client_pool = Config().clients.clients_requesting_deletion

        if self.client_id in client_pool and self.current_round <= previous_round:
            logging.info(
                "[%s] Unlearning sampler deployed: %s%% of the samples were deleted.",
                self,
                Config().clients.deleted_data_ratio * 100)

            if not hasattr(Config().data,
                           'reload_data') or Config().data.reload_data:
                logging.info("[%s] Loading the dataset.", self)
                self.load_data()

            self.sampler = unlearning_iid.Sampler(self.datasource,
                                                  self.client_id, False)

        self.previous_round[self.client_id] = self.current_round
