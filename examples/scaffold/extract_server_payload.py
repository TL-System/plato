"""Processor for extracting additional information attached to the server payload."""

import logging
from typing import List

from plato.processors import base


class Processor(base.Processor):
    """
    Implements a processor for clients to extract additional information attached to the
    received server payload
    """

    def process(self, data: List) -> List:

        self.trainer.additional_data = data[1]

        if data[1] is not None:
            logging.info(
                "[Client #%d] Extracted information attached to payload.",
                self.trainer.client_id,
            )
        return data[0]
