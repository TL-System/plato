"""
Dummy processor which can be used to test customize processor feature.
"""

import logging
from typing import Any

from plato.processors import base


class DummyProcessor(base.Processor):
    """A dummy processor to be instantiated and inserted to the list of processors at runtime."""

    def __init__(self, client_id, current_round, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id
        self.current_round = current_round

    def process(self, data: Any) -> Any:
        logging.info(
            "[Client #%s] Customized dummmy processor is activated at round %s.",
            self.client_id,
            self.current_round,
        )
        return data
