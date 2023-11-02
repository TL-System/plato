"""
A customized client for Knot, a clustered aggregation mechanism designed for
federated unlearning.
"""
import logging

from plato.config import Config
from plato.clients import simple

import unlearning_iid


class Client(simple.Client):
    """Knot: a clustered aggregation mechanism designed for federated unlearning."""

    def process_server_response(self, server_response):
        """
        If a client requested data deletion, replace its sampler accordingly in the
        retraining phase.
        """
        client_pool = Config().clients.clients_requesting_deletion

        if self.client_id in client_pool and "rollback_round" in server_response:
            logging.info(
                "[%s] Unlearning sampler deployed: %s%% of the samples were deleted.",
                self,
                Config().clients.deleted_data_ratio * 100,
            )

            if not hasattr(Config().data, "reload_data") or Config().data.reload_data:
                logging.info("[%s] Loading the dataset.", self)
                self._load_data()

            self.sampler = unlearning_iid.Sampler(
                self.datasource, self.client_id, False
            )
