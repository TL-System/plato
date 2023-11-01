"""
Customize client callbacks for assigning the modules of client's local model
to the received payload.
"""
import logging
import os
from typing import Any

from pflbases.client_callbacks import base_callbacks

from plato.config import Config
from plato.processors import base


class PayloadCompletionProcessor(base.Processor):
    """
    A processor relying on the hyper-parameter `local_module_names` to complete parameters of payload
    with the loaded local model, which is the updated global model in the previous round.
    """

    def __init__(self, algorithm, **kwargs) -> None:
        super().__init__(**kwargs)
        self.algorithm = algorithm

    def process(self, data: Any) -> Any:
        """Processing the received payload by replacing the local layers with a client's own."""
        local_module_names = Config().algorithm.local_module_names

        # Load the previously saved local model
        filename = f"client_{self.trainer.client_id}_local_model.pth"
        location = Config().params["checkpoint_path"]

        if os.path.exists(os.path.join(location, filename)):
            self.trainer.load_model(filename, location=location)

        # Extract desired local modules
        local_layers = self.algorithm.extract_local_weights(local_module_names)

        # Replace the corresponding layers in the received global model with the local counterparts
        data.update(local_layers)

        logging.info(
            "[Client #%d] Replaced portions of the global model with local layers.",
            self.trainer.client_id,
        )

        return data


class PayloadCompletionCallback(base_callbacks.ClientPayloadCallback):
    """
    A client callback for processing payload by assigning parameters of the local
    model to it.
    """

    def on_inbound_received(self, client, inbound_processor):
        """Complete the recevied payload with the local updated model,
        which is the locally updated global model in the previous round."""
        super().on_inbound_received(client, inbound_processor)

        inbound_processor.processors.append(
            PayloadCompletionProcessor(
                trainer=client.trainer,
                algorithm=client.algorithm,
                name="LocalPayloadCompletionProcessor",
            )
        )
