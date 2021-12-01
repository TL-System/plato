"""
Implements a Processor for converting MistNet features from PyTorch tensors to numpy ndarrays.
"""
import logging
from typing import Any

from plato.processors import base


class Processor(base.Processor):
    """
    Implements a Processor for converting MistNet features from PyTorch tensors to numpy ndarrays.
    This is used only by MistNet clients at this time.
    """
    def __init__(self, client_id=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id

    def process(self, data: Any) -> Any:
        """
        Converts MistNet features from PyTorch tensors to numpy ndarrays.
        """
        feature_dataset = []

        for logit, target in data:
            feature_dataset.append(
                (logit.detach().cpu().numpy(), target.detach().cpu().numpy()))

        logging.info(
            "[Client #%d] Features converted from PyTorch tensors to ndarrays.",
            self.client_id)

        return feature_dataset
