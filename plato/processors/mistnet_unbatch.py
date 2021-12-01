"""
Implements a Processor for unbatching MistNet PyTorch features into the dataset form.
"""
import logging
from typing import Any

import numpy as np
from plato.processors import base


class Processor(base.Processor):
    """
    Implements a Processor for unbatching MistNet PyTorch features into the dataset form.
    """
    def __init__(self, client_id=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id

    def process(self, data: Any) -> Any:
        """
        Implements a Processor for unbatching MistNet PyTorch features into the dataset form.
        """
        feature_dataset = []

        for logits, targets in data:
            for i in np.arange(logits.shape[0]):  # each sample in the batch
                feature_dataset.append((logits[i], targets[i]))

        logging.info("[Client #%d] Features extracted from %s examples.",
                     self.client_id, len(feature_dataset))

        return feature_dataset
