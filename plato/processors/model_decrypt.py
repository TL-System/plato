"""
A processor that decrypts model weights of MaskCrypt.
"""
from typing import Any

import torch
from plato.processors import model
from plato.utils import homo_enc


class Processor(model.Processor):
    """
    A processor that decrypts model tensors
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.context = homo_enc.get_ckks_context()
        weight_shapes = {}
        para_nums = {}
        extract_model = self.trainer.model.cpu().state_dict()

        for key in extract_model.keys():
            weight_shapes[key] = extract_model[key].size()
            para_nums[key] = torch.numel(extract_model[key])

        self.weight_shapes = weight_shapes
        self.para_nums = para_nums

    def process(self, data: Any) -> Any:
        """Deserialize and decrypt the model weights."""
        deserialized_weights = homo_enc.deserialize_weights(data, self.context)

        output = homo_enc.decrypt_weights(
            deserialized_weights, self.weight_shapes, self.para_nums
        )

        return output
