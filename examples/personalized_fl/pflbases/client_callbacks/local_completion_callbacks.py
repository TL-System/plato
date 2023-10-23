"""
Customize client callbacks for assigning the modules of client's local model
to the received payload.
"""

import logging
from typing import Any

from plato.config import Config
from plato.processors import base

from pflbases.client_callbacks import base_callbacks


class PayloadCompletionProcessor(base.Processor):
    """A processor relying on the hyper-parameter `completion_modules_name`
    to complete parameters of payload with the loaded local model, which
    is the updated global model in the previous round."""

    def __init__(self, current_round, algorithm, **kwargs) -> None:
        super().__init__(**kwargs)
        self.current_round = current_round
        self.algorithm = algorithm

    def process(self, data: Any) -> Any:
        """Processing the received payload by assigning modules of local model of
        each client."""

        # extract the `completion_modules_name` of the model head
        assert hasattr(Config().algorithm, "completion_modules_name")

        completion_modules_name = Config().algorithm.completion_modules_name
        local_model_modules = self.algorithm.extract_weights(
            model=self.trainer.model, modules_name=completion_modules_name
        )
        logging.info(
            "[Client #%d] Extracted modules: %s from its loaded local model.",
            self.trainer.client_id,
            self.algorithm.extract_modules_name(list(local_model_modules.keys())),
        )

        data.update(local_model_modules)

        logging.info(
            "[Client #%d] Completed the payload with extracted modules.",
            self.trainer.client_id,
        )

        return data


class ClientModelLocalCompletionCallback(base_callbacks.ClientPayloadCallback):
    """
    A client callback for processing payload by assigning parameters of the local
    model to it.
    """

    def on_inbound_received(self, client, inbound_processor):
        """Completing the recevied payload with the local updated model,
        which is the locally updated global model in the previous round."""
        super().on_inbound_received(client, inbound_processor)

        inbound_processor.processors.append(
            PayloadCompletionProcessor(
                current_round=client.current_round,
                trainer=client.trainer,
                algorithm=client.algorithm,
            )
        )
