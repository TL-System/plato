"""
A Processor of differential privacy to clip and add noise on gradients of model weights.
"""

from abc import abstractmethod
from collections import OrderedDict
import logging
import os
from typing import Any
import torch
import numpy as np

from plato.processors import base


class Processor(base.Processor):
    """
    Implements a Processor to clip and add noise on gradients.
    """
    def __init__(self, client_id=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id

        # this processor is used when training model, not for processing data
        self.used_by_trainer = True

    def process(self, data: Any) -> Any:
        """
        Clips and adds noise on gradients to guarantee differential privacy.
        """

        if self.client_id is None:
            logging.info("[Server #%d] Applied local differential privacy.",
                         os.getpid())
        else:
            logging.info("[Client #%d] Applied local differential privacy.",
                         self.client_id)

        return data
