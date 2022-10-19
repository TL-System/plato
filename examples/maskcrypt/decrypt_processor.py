"""
A processor that decrypts model weights tensors.
"""

from typing import Any
import torch

from plato.processors import model
import encrypt_utils
from plato.config import Config


class Processor(model.Processor):
    """
    A processor that decrypts model tensors
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.context = encrypt_utils.get_ckks_context()
        weight_shapes = {}
        para_nums = {}
        extract_model = self.trainer.model.cpu().state_dict()

        for key in extract_model.keys():
            weight_shapes[key] = extract_model[key].size()
            para_nums[key] = torch.numel(extract_model[key])
        self.weight_shapes = weight_shapes
        self.para_nums = para_nums

    def process(self, data: Any) -> Any:
        encrypt_utils.update_est(Config(), self.client_id, data)
        deserialized_weights = encrypt_utils.deserialize_weights(data, self.context)
        if self.client_id:
            output = encrypt_utils.decrypt_weights(
                deserialized_weights, self.weight_shapes, self.para_nums
            )
        else:
            output = deserialized_weights
        return output
