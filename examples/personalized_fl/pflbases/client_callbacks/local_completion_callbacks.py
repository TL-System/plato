"""
Customize client callbacks for assigning the modules of client's local model
to the received payload.
"""
import os
import logging
from typing import Any

from plato.config import Config
from plato.processors import base

from pflbases.client_callbacks import base_callbacks
from pflbases.fedavg_partial import Algorithm


class PayloadCompletionProcessor(base.Processor):
    """A processor relying on the hyper-parameter `local_module_names`
    to complete parameters of payload with the loaded local model, which
    is the updated global model in the previous round."""

    def __init__(self, trainer, **kwargs) -> None:
        super().__init__(**kwargs)

        self.trainer = trainer

    def process(self, data: Any) -> Any:
        """Processing the received payload by assigning modules of local model of
        each client."""

        # Extract the `local_module_names` of the model head.
        assert hasattr(Config().algorithm, "local_module_names")
        local_module_names = Config().algorithm.local_module_names

        # Load the previously saved local model
        filename = f"client_{self.trainer.client_id}_local_model.pth"
        location = Config().params["checkpoint_path"]

        if os.path.exists(os.path.join(location, filename)):
            self.trainer.load_model(filename, location=location)

        model_modules = self.trainer.model.cpu().state_dict()
        logging.info(
            "[Client #%d] The local model contains: %s.",
            self.trainer.client_id,
            Algorithm.extract_module_names(list(model_modules.keys())),
        )

        # Extract desired local modules
        local_modules = Algorithm.get_module_weights(
            model_parameters=model_modules, module_names=local_module_names
        )
        # Update the payload with the local modules
        data.update(local_modules)

        logging.info(
            "[Client #%d] Completed the payload by extracting local modules: %s.",
            self.trainer.client_id,
            Algorithm.extract_module_names(list(local_modules.keys())),
        )

        return data


class SaveLocalModelProcessor(base.Processor):
    """
    A processor for clients to always save the trained local model.
    """

    def __init__(self, client_id, trainer, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id
        self.trainer = trainer

    def process(self, data: Any):
        """Save the local model before sending the data."""
        filename = f"client_{self.client_id}_local_model.pth"
        location = Config().params["checkpoint_path"]
        self.trainer.save_model(filename, location=location)

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
                name="LocalPayloadCompletionProcessor",
            )
        )

    def on_outbound_ready(self, client, report, outbound_processor):
        """Save local model of the client."""
        send_payload_processor = SaveLocalModelProcessor(
            client_id=client.client_id,
            trainer=client.trainer,
            name="SaveLocalModelProcessor",
        )
        outbound_processor.processors.insert(0, send_payload_processor)
