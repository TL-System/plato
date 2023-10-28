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

    def __init__(self, algorithm, **kwargs) -> None:
        super().__init__(**kwargs)
        self.algorithm = algorithm

    def process(self, data: Any) -> Any:
        """Processing the received payload by assigning modules of local model of
        each client."""

        # extract the `completion_modules_name` of the model head
        assert hasattr(Config().algorithm, "completion_modules_name")

        completion_modules_name = Config().algorithm.completion_modules_name
        local_model_modules = self.trainer.model.cpu().state_dict()
        logging.info(
            "[Client #%d] Loaded the local model containing modules: %s.",
            self.trainer.client_id,
            self.algorithm.extract_modules_name(list(local_model_modules.keys())),
        )

        local_completion_modules = self.algorithm.get_target_weights(
            model_parameters=local_model_modules, modules_name=completion_modules_name
        )

        data.update(local_completion_modules)

        logging.info(
            "[Client #%d] Completed the payload with extracted modules: %s.",
            self.trainer.client_id,
            self.algorithm.extract_modules_name(list(local_completion_modules.keys())),
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
                trainer=client.trainer,
                algorithm=client.algorithm,
                name="LocalPayloadCompletionProcessor",
            )
        )
