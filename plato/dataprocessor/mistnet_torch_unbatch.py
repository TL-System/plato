"""
DataProcessor for unbatching mistnet pytorch features into dataset form.
Only used for features in mistnet in pytorch.
"""
from typing import Any
import logging

import numpy as np

from plato.dataprocessor import base


class DataProcessor(base.DataProcessor):
    """
    DataProcessor class.
    Base DataProcessor class implementation do nothing on the data.
    """
    def __init__(self, *args, client_id=None, **kwargs) -> None:
        """Constructor for DataProcessor"""
        self.client_id = client_id

    def process(self, data: Any) -> Any:
        """
        Data processing implementation.
        Implement this method while inheriting the class.
        """

        feature_dataset = []

        for logits, targets in data:
            for i in np.arange(logits.shape[0]):  # each sample in the batch
                feature_dataset.append((logits[i], targets[i]))

        logging.info("[Client #%d] Features extracted from %s examples.",
                     self.client_id, len(feature_dataset))

        return feature_dataset
