"""
A Processor of differential privacy to clip and add noise on gradients of model weights.
"""

import logging
import os
from typing import Any
from opacus.privacy_engine import PrivacyEngine

from plato.processors import base
from plato.config import Config


class Processor(base.Processor):
    """
    Implements a Processor to clip and add noise on gradients.
    """
    def __init__(self, client_id=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id

        # this processor is used when training a model, not for processing data
        self.used_by_trainer = True

    def configure(self, model, optimizer, train_loader, epochs):
        """
        Configures the privacy engine to apply differential privacy.
        """
        privacy_engine = PrivacyEngine(accountant='rdp', secure_mode=False)
        # accountant: Accounting mechanism. Currently supported:
        #         - rdp (:class:`~opacus.accountants.RDPAccountant`)
        #         - gdp (:class:`~opacus.accountants.GaussianAccountant`)

        return privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=Config().algorithm.dp_epsilon,
            target_delta=Config().algorithm.dp_delta if hasattr(
                Config().algorithm, 'dp_delta') else 1e-5,
            epochs=epochs,
            max_grad_norm=100,
        )

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
