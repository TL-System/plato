"""
Callback for adding the divergence rate to the payload.
"""

import calibre_processor

from plato.callbacks.client import ClientCallback


class CalibreCallback(ClientCallback):
    """
    A client callback that adds the divergence rate computed locally to the
    payload sent to the server.
    """

    def on_outbound_ready(self, client, report, outbound_processor):
        """
        Insert a AddDivergenceRateProcessor to the list of outbound processors.
        """
        send_payload_processor = calibre_processor.AddDivergenceRateProcessor(
            client_id=client.client_id,
            trainer=client.trainer,
            name="AddDivergenceRateProcessor",
        )

        outbound_processor.processors.insert(0, send_payload_processor)
