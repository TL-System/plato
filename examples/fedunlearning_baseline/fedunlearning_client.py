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

    def process_server_response(self, server_response):
        if server_response['retrain_phase'] or self.current_round > Config(
        ).clients.data_deletion_round:
            if self.client_id in Config().clients.client_requesting_deletion:
                logging.info(
                    "[%s] Unlearning sampler deployed: %s%% of the samples were deleted.",
                    self,
                    Config().clients.deleted_data_ratio * 100)

                if (hasattr(Config().data, 'reload_data')
                        and Config().data.reload_data) or not self.data_loaded:
                    logging.info("[%s] Loading the dataset.", self)
                    self.load_data()

                self.sampler = unlearning_iid.Sampler(self.datasource,
                                                      self.client_id, False)
