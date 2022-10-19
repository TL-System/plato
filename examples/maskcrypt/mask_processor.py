"""
A processor that computes the encryption mask.
"""

import pickle
import random
from turtle import st
import logging
import time
from typing import Any

import torch
import tenseal as ts
from plato.config import Config

from plato.processors import base
import encrypt_utils as homo_enc


class Processor(base.Processor):
    def __init__(self, client, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gradient = client.trainer.gradient

    def process(self, data: Any) -> Any:
        selected_mask = self.compute_mask(data)
        return selected_mask

    def get_exposed_weights(self):
        model_name = Config().trainer.model_name

        est_filename = f"{self.checkpoint_path}/{self.attack_prep_dir}/{model_name}_est_{self.client_id}.pth"
        return homo_enc.get_est(est_filename)

    def compute_mask(self, latest_weights):
        exposed_flat = self.get_exposed_weights()
        device = exposed_flat.device

        latest_flat = torch.cat(
            [
                torch.flatten(latest_weights[name])
                for _, name in enumerate(latest_weights)
            ]
        )
        # Store the plain model weights
        plain_filename = f"{self.checkpoint_path}/{self.attack_prep_dir}/{Config().trainer.model_name}_plain_{self.client_id}.pth"
        with open(plain_filename, "wb") as plain_file:
            pickle.dump(latest_flat, plain_file)

        if self.random_mask:
            rand_mask = random.sample(
                range(len(exposed_flat)), int(self.encrypt_ratio * len(exposed_flat))
            )
            return torch.tensor(rand_mask)
        gradient_list = [
            torch.flatten(gradients[name]).to(device)
            for _, name in enumerate(gradients)
        ]
        grad_flat = torch.cat(gradient_list)

        delta = exposed_flat - latest_flat
        product = delta * grad_flat

        _, indices = torch.sort(product, descending=True)
        mask_len = int(self.encrypt_ratio * len(indices))

        return indices[:mask_len].clone().detach()
