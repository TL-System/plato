"""
Callback for adding the divergence rate to the payload.
"""

import logging
from typing import OrderedDict

import torch

from plato.callbacks.client import ClientCallback
from plato.processors import base


class AddDivergenceRateProcessor(base.Processor):
    """
    Implements a processor for adding the divergence rate to the payload.
    """

    def __init__(self, client_id, trainer, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id
        self.trainer = trainer

    def process(self, data: OrderedDict):
        save_path = self.trainer.get_divergence_filepath()

        divergence_rate = torch.load(save_path)

        data = [data, divergence_rate]

        logging.info(
            "[Client #%d] Divergence Rate attached to payload.", self.client_id
        )
        return data


class CalibreCallback(ClientCallback):
    """
    A client callback that adds the divergence rate computed locally to the
    payload sent to the server.
    """

    def on_outbound_ready(self, client, report, outbound_processor):
        """
        Insert a AddDivergenceRateProcessor to the list of outbound processors.
        """
        send_payload_processor = AddDivergenceRateProcessor(
            client_id=client.client_id,
            trainer=client.trainer,
            name="AddDivergenceRateProcessor",
        )

        outbound_processor.processors.insert(0, send_payload_processor)

        logging.info(
            "[%s] List of outbound processors: %s.",
            client,
            outbound_processor.processors,
        )
