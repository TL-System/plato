"""
Customize the inbound and outbound processors for MaskCrypt clients through callbacks.
"""

from typing import Any

import maskcrypt_utils

from plato.callbacks.client import ClientCallback
from plato.config import Config
from plato.processors import base, model_decrypt, model_encrypt


class ModelEstimateProcessor(base.Processor):
    """
    A client processor used to track the exposed model weights so far.
    """

    def __init__(self, client_id, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client_id = client_id

    def process(self, data: Any) -> Any:
        maskcrypt_utils.update_est(Config(), self.client_id, data)

        return data


class MaskCryptCallback(ClientCallback):
    """
    A client callback that dynamically inserts encrypt and decrypt processors.
    """

    def on_inbound_received(self, client, inbound_processor):
        current_round = client.current_round
        if current_round % 2 != 0:
            # Update the exposed model weights from new global model
            inbound_processor.processors.append(
                ModelEstimateProcessor(client_id=client.client_id)
            )

            # Server sends model weights in odd rounds, add decrypt processor
            inbound_processor.processors.append(
                model_decrypt.Processor(
                    client_id=client.client_id,
                    trainer=client.trainer,
                    name="model_decrypt",
                )
            )

    def on_outbound_ready(self, client, report, outbound_processor):
        current_round = client.current_round

        if current_round % 2 == 0:
            # Clients send model weights to server in even rounds, add encrypt processor
            outbound_processor.processors.append(
                model_encrypt.Processor(
                    mask=client.final_mask,
                    client_id=client.client_id,
                    trainer=client.trainer,
                    name="model_encrypt",
                )
            )

            # Update the exposed model weights after encryption
            outbound_processor.processors.append(
                ModelEstimateProcessor(client_id=client.client_id)
            )
