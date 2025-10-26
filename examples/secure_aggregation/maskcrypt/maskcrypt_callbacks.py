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
            processors = inbound_processor.processors
            processors[:] = [
                proc
                for proc in processors
                if getattr(proc, "name", None)
                not in {"MaskCryptModelEstimate", "model_decrypt"}
            ]
            decode_index = next(
                (
                    idx
                    for idx, proc in enumerate(processors)
                    if getattr(proc, "name", None) == "safetensor_decode"
                ),
                -1,
            )

            estimate_processor = ModelEstimateProcessor(
                client_id=client.client_id, name="MaskCryptModelEstimate"
            )
            decrypt_processor = model_decrypt.Processor(
                client_id=client.client_id,
                trainer=client.trainer,
                name="model_decrypt",
            )

            insert_at = decode_index + 1 if decode_index >= 0 else len(processors)
            processors.insert(insert_at, estimate_processor)
            processors.insert(insert_at + 1, decrypt_processor)
        else:
            inbound_processor.processors[:] = [
                proc
                for proc in inbound_processor.processors
                if getattr(proc, "name", None)
                not in {"MaskCryptModelEstimate", "model_decrypt"}
            ]

    def on_outbound_ready(self, client, report, outbound_processor):
        current_round = client.current_round

        if current_round % 2 == 0:
            processors = outbound_processor.processors
            processors[:] = [
                proc
                for proc in processors
                if getattr(proc, "name", None)
                not in {"model_encrypt", "MaskCryptModelEstimate"}
            ]
            encode_index = next(
                (
                    idx
                    for idx, proc in enumerate(processors)
                    if getattr(proc, "name", None) == "safetensor_encode"
                ),
                len(processors),
            )

            encrypt_processor = model_encrypt.Processor(
                mask=client.final_mask,
                client_id=client.client_id,
                trainer=client.trainer,
                name="model_encrypt",
            )
            estimate_processor = ModelEstimateProcessor(
                client_id=client.client_id, name="MaskCryptModelEstimate"
            )

            processors.insert(encode_index, encrypt_processor)
            processors.insert(encode_index + 1, estimate_processor)
        else:
            outbound_processor.processors[:] = [
                proc
                for proc in outbound_processor.processors
                if getattr(proc, "name", None)
                not in {"model_encrypt", "MaskCryptModelEstimate"}
            ]
