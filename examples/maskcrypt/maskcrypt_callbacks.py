import encrypt_processor
import decrypt_processor

from plato.callbacks.client import ClientCallback


class MaskCryptCallback(ClientCallback):
    """
    A client callback that dynamically inserts processors into the current list of inbound
    processors.
    """

    def on_inbound_received(self, client, inbound_processor):
        current_round = client.current_round
        if current_round % 2 != 0:
            # Server sends model weights in odd rounds, add decrypt processor
            inbound_processor.processors.append(
                decrypt_processor.Processor(
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
                encrypt_processor.Processor(
                    mask=client.final_mask,
                    client_id=client.client_id,
                    trainer=client.trainer,
                    name="model_encrypt",
                )
            )
