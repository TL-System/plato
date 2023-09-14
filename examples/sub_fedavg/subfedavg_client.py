"""
A federated learning client using pruning in Sub-FedAvg.
"""

import logging

from plato.clients import simple


class Client(simple.Client):
    """
    A federated learning client prunes its update before sending out.
    """

    async def _train(self):
        """The training process on a Sub-FedAvg client."""

        # Perform model training
        self._report, weights = await super()._train()

        logging.info("[Client #%d] Trained with Sub-FedAvg algorithm.", self.client_id)

        return self._report, weights
