"""
Customize client callbacks to complete the received payload with third-part 
parameters. For instance, when payload contains the body of one model, the head 
of the loaded personalized model will be used to complete the payload.
"""


import logging
from typing import Any

from plato.config import Config
from plato.processors import base

from bases.client_callbacks import base_callbacks


class PayloadCompletionProcessor(base.Processor):
    """A processor relying on the hyper-parameter `completion_modules_name`
    to complete parameters of payload with the loaded personalized model."""

    def __init__(self, algorithm, **kwargs) -> None:
        super().__init__(**kwargs)
        self.algorithm = algorithm

    def process(self, data: Any) -> Any:
        """Processing the received payload by assigning modules of personalized model
        if provided."""

        # extract the `completion_modules_name` of the personalized model head
        completion_modules_name = Config().algorithm.completion_modules_name
        model_modules = self.algorithm.extract_weights(
            model=self.trainer.personalized_model, modules_name=completion_modules_name
        )
        logging.info(
            "[Client #%d] Extracted modules: %s from its loaded personalized model.",
            self.trainer.client_id,
            self.algorithm.extract_modules_name(list(model_modules.keys())),
        )

        data.update(model_modules)

        logging.info(
            "[Client #%d] Completed the payload with extracted modules.",
            self.trainer.client_id,
        )

        return data


class ClientModelCompletionCallback(base_callbacks.ClientPayloadCallback):
    """
    A client callback for FedBABU approach to process the received payload.
    """

    def on_inbound_received(self, client, inbound_processor):
        """Completing the recevied payload with the personalized model."""
        super().on_inbound_received(client, inbound_processor)

        inbound_processor.processors.append(
            PayloadCompletionProcessor(
                trainer=client.trainer, algorithm=client.algorithm
            )
        )
