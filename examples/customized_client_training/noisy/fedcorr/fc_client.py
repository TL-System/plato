import logging
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

        self.server_id = os.getpid()
        self.cache_root = os.path.expanduser("~/.cache")

    async def inbound_processed(self, processed_inbound_payload):
        stage = processed_inbound_payload["stage"]
        original_payload = processed_inbound_payload["payload"]
        self.trainer.set_server_id(self.server_id)
        self.trainer.set_stage(stage)

        # Clean clients don't relabel
        noisy_clients = processed_inbound_payload["noisy_clients"]
        if noisy_clients and self.client_id not in noisy_clients:
            label_file = f"{self.server_id}-client-{self.client_id}-label-updates.pt"
            label_file = os.path.join(self.cache_root, label_file)
            if os.path.exists(label_file):
                os.remove(label_file)

        self.eval_and_setup_pseudo_labels()
        
        logging.warn(f"{self} STAGE {stage}")
        if stage == 1:
            report, model_weights = await super().inbound_processed(original_payload)
            LID_file = f"{self.server_id}_LID_client_{self.client_id}.pt"
            LID_file = os.path.join(self.cache_root, LID_file)
            LID_client = torch.load(LID_file)

            outbound_payload = {
                "model_weights": model_weights,
                "LID_client": LID_client,
                "client_id": self.client_id
            }
            return report, outbound_payload
        
        elif stage == 2:
            if self.client_id in noisy_clients:
                self.trainer.set_relabel_only()
            report, model_weights = await super().inbound_processed(original_payload)
            return report, model_weights
        elif stage == 3:
            return await super().inbound_processed(original_payload)

    def eval_and_setup_pseudo_labels(self,):
        logging.info(f"[{self}] Evaluate predicted pseudo labels.")
        client_id = self.client_id
        client_indices = self.sampler.get().indices
        self.datasource.eval_pseudo_acc(client_id, client_indices)

        logging.info(f"[{self}] Read pseudo labels from file.")
        self.datasource.setup_client_datasource(self.client_id)