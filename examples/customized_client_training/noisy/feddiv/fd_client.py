import os
import random
import time
import torch
import pickle

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        self.local_filter = None
        self.server_id = os.getpid()
        self.cache_root = os.path.expanduser("~/.cache")

    async def inbound_processed(self, processed_inbound_payload):
        warm_up = processed_inbound_payload["warm_up"] if "warm_up" in processed_inbound_payload else None
        original_payload = processed_inbound_payload["payload"]
        self.trainer.warm_up = warm_up
        if warm_up:
            # Conduct local training and compute encryption mask after that
            report, model_weights = await super().inbound_processed(original_payload)
            return report, model_weights
        else:
            global_filter_stat = processed_inbound_payload["filter"]
            self.trainer.set_local_filter(global_filter_stat, self.sampler)
            self.trainer.set_server_id(self.server_id)

            # Conduct local training and compute encryption mask after that
            report, model_weights = await super().inbound_processed(original_payload)

            # Load filter updates
            filter_update_file = f"{self.server_id}_filter_{self.client_id}.pt"
            filter_update_file = os.path.join(self.cache_root, filter_update_file)
            filter_updates = torch.load(filter_update_file)

            outbound_payload = {
                "model_weights": model_weights,
                "filter_updates": filter_updates,
                "data_size": len(self.sampler.get().indices),
            }

            return report, outbound_payload
