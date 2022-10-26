"""
A processor that encrypts model weights tensors.
"""

import logging
import time
import torch

from typing import Any
from plato.config import Config
from plato.processors import model

import encrypt_utils


class Processor(model.Processor):
    def __init__(self, mask=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.context = encrypt_utils.get_ckks_context()
        self.mask = mask

        para_nums = {}
        extract_model = self.trainer.model.cpu().state_dict()
        for key in extract_model.keys():
            para_nums[key] = torch.numel(extract_model[key])
        self.para_nums = para_nums

    def process(self, data: Any) -> Any:
        logging.info(
            "[Client #%d] Applying a processor that encrypts the model.", self.client_id
        )
        start_time = time.time()
        encrypted_weights = encrypt_utils.encrypt_weights(
            data,
            serialize=True,
            context=self.context,
            encrypt_indices=self.mask,
        )

        logging.info(f"Encryption Time: {time.time() - start_time}")

        encrypt_utils.update_est(Config(), self.client_id, encrypted_weights)
        return encrypted_weights
