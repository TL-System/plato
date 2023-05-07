"""
Customize the 

"""

import logging
from typing import Any

from plato.config import Config
from plato.processors import base

from bases import client_callbacks


class ModelCompletionProcessor(base.Processor):
    """A processor aiming to complete parameters of payload with the
    loaded personalized model."""

    def __init__(self, algorithm, **kwargs) -> None:
        super().__init__(**kwargs)
        self.algorithm = algorithm

    def process(self, data: Any) -> Any:
        """Processing the received payload by assigning the head of personalized model
        if provided."""
        # logging what modules have been received
        data = super().process(data)

        # extract the `head` of the personalized model head
        head_modules_name = Config().algorithm.head_modules_name
        model_head_params = self.algorithm.extract_weights(
            model=self.trainer.personalized_model, modules_name=head_modules_name
        )
        logging.info(
            "[Client #%d] Extracted head modules: %s from its loaded personalized model.",
            self.trainer.client_id,
            self.algorithm.extract_modules_name(list(model_head_params.keys())),
        )

        data.update(model_head_params)

        logging.info(
            "[Client #%d] Combined head modules to received modules.",
            self.trainer.client_id,
        )

        return data


class ClientModelCallback(client_callbacks.ClientModelCallback):
    """
    A client callback for FedBABU approach to process the received payload.
    """

    def on_inbound_received(self, client, inbound_processor):
        """Completing the recevied payload with the personalized model."""
        super().on_inbound_received(client, inbound_processor)

        inbound_processor.processors.append(
            ModelCompletionProcessor(trainer=client.trainer, algorithm=client.algorithm)
        )
