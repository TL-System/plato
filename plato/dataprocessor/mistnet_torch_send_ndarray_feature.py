"""
DataProcessor for converting mistnet pytorch features into ndarray.
Only used for features in mistnet in pytorch.
"""
from typing import Any
import logging

import numpy as np

from plato.dataprocessor import base


class DataProcessor(base.DataProcessor):
    """
    DataProcessor class.
    DataProcessor for converting mistnet pytorch features into ndarray.
    """
    def __init__(self, *args, client_id=None, **kwargs) -> None:
        """Constructor for DataProcessor"""
        self.client_id = client_id

    def process(self, data: Any) -> Any:
        """
        Data processing implementation.
        Converting mistnet pytorch features into ndarray.
        """

        feature_dataset = []

        for logit, target in data:
            feature_dataset.append(
                (logit.detach().cpu().numpy(), target.detach().cpu().numpy()))

        logging.info("[Client #%d] Features converted to ndarray for sending.",
                     self.client_id)

        return feature_dataset
