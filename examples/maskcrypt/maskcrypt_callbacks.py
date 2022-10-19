from plato.callbacks.client import ClientCallback
import mask_processor
import encrypt_processor
import decrypt_processor


class MaskCryptCallback(ClientCallback):
    """
    A client callback that dynamically inserts processors into the current list of inbound
    processors.
    """

    def on_inbound_process(self, client, inbound_processor):
        current_round = client.current_round
        if current_round % 2 != 0:
            inbound_processor.processors.append(
                decrypt_processor.Processor(client_id=client.client_id, name="model_decrypt")
            )

    def on_outbound_process(self, client, outbound_processor):
        current_round = client.current_round
        if current_round % 2 == 0:
            outbound_processor.processors.append(
                encrypt_processor.Processor(client_id=client.client_id, name="model_encrypt")
            )
        else:
            outbound_processor.processors.append(
                mask_processor.Processor(
                    client=client, client_id=client.client_id, name="mask_generator"
                )
            )
