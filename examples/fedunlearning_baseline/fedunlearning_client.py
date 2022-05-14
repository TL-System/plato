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


def decode_config_with_comma(target_string):
    """ Split the input target_string as int by comma. """
    if isinstance(target_string, int):
        return [target_string]
    else:
        return list(map(int, target_string.split(", ")))


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

        # Recording which clients reach the delete conditions. key: ids, value: if it needs deletion
        self.clients_need_to_be_deleted = {}

    def process_server_response(self, server_response):
        if server_response['retrain_phase'] or self.current_round > Config(
        ).clients.data_deletion_round:
            client_requesting_deletion_ids = decode_config_with_comma(
                Config().clients.client_requesting_deletion)

            for client_requesting_deletion_id in client_requesting_deletion_ids:
                self.clients_need_to_be_deleted[client_requesting_deletion_id] = True

            if self.client_id in client_requesting_deletion_ids:

                if self.clients_need_to_be_deleted[self.client_id] is True:
                    logging.info(
                        "[%s] Unlearning sampler deployed: %s%% of the samples were deleted.",
                        self,
                        Config().clients.deleted_data_ratio * 100)

                    if (hasattr(Config().data, 'reload_data') and
                            Config().data.reload_data) or not self.data_loaded:
                        self.load_data()

                    self.sampler = unlearning_iid.Sampler(
                        self.datasource, self.client_id, False)
                else:
                    pass
            else:
                pass
        else:
            pass
