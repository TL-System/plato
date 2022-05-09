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
        self.current_round = 0
        self.testset_sampler = None

    def process_server_response(self, server_response):
        if self.current_round == Config().clients.data_deleted_round:
            logging.info(
                "[%s] Unlearning sampler deployed: %s of the samples were deleted.",
                self,
                Config().clients.deleted_data_ratio)

            self.sampler = unlearning_iid.Sampler(self.datasource,
                                                  self.client_id, False)
