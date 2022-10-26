"""
A processor that encrypts model weights in MaskCrypt.
"""
import logging
import torch
import encrypt_utils

from typing import Any
from plato.config import Config
from plato.processors import model


class Processor(model.Processor):
    """
    A processor that encrypts model weights with given encryption mask.
    """

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
            "[Client #%d] Encrypt the model weights with given encryption mask.",
            self.client_id,
        )
        encrypted_weights = encrypt_utils.encrypt_weights(
            data,
            serialize=True,
            context=self.context,
            encrypt_indices=self.mask,
        )

        encrypt_utils.update_est(Config(), self.client_id, encrypted_weights)
        return encrypted_weights
