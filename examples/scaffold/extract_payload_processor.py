"""Processor for extracting additional information attached to the server payload."""

import logging
from typing import List

from plato.processors import base


class ExtractPayloadProcessor(base.Processor):
    """
    A processor for clients to extract additional information attached to the
    received server payload.
    """

    def __init__(self, client_id, current_round, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id
        self.current_round = current_round

    def process(self, data: List) -> List:

        self.trainer.additional_data = data[1]

        if data[1] is not None:
            logging.info(
                "[Client #%d] Extracted information attached to payload.",
                self.trainer.client_id,
            )
        return data[0]
