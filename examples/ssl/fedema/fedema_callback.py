"""
The ClientCallback used by FedEMA to add on the FedEMA processor.
"""

import fedema_processor

from plato.callbacks.client import ClientCallback


class FedEMACallback(ClientCallback):
    """
    A client callback that dynamically compute the divergence between the received model
    and the local model.
    """

    def on_inbound_received(self, client, inbound_processor):
        """
        Insert an GlobalLocalDivergenceProcessor to the list of inbound processors.
        """
        extract_payload_processor = fedema_processor.GlobalLocalDivergenceProcessor(
            trainer=client.trainer,
            name="GlobalLocalDivergenceProcessor",
        )
        inbound_processor.processors.insert(0, extract_payload_processor)
