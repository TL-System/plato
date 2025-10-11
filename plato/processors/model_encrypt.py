"""
A processor that encrypts model weights in MaskCrypt.
"""

import logging
from typing import Any

import torch

from plato.processors import model
from plato.utils import homo_enc


class Processor(model.Processor):
    """
    A processor that encrypts model weights with given encryption mask.
    """

    def __init__(self, mask=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.context = homo_enc.get_ckks_context()
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

        encrypted_weights = homo_enc.encrypt_weights(
            data,
            serialize=True,
            context=self.context,
            indices=self.mask,
        )

        return encrypted_weights
