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
        super().__init__(model, datasource, algorithm, trainer)
        self.previous_round = {}
        self.testset_sampler = None

    def process_server_response(self, server_response):
        if self.client_id in self.previous_round:
            previous_round = self.previous_round[self.client_id]
        else:
            previous_round = 0

        if self.current_round == 1 and self.current_round <= previous_round:
            logging.info(
                "[%s] Unlearning sampler deployed: %s%% of the samples were deleted.",
                self,
                Config().clients.deleted_data_ratio * 100)

            self.sampler = unlearning_iid.Sampler(self.datasource,
                                                  self.client_id, False)

        self.previous_round[self.client_id] = self.current_round
