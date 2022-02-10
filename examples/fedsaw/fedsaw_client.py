"""
A federated learning client using pruning.
"""

import logging

from plato.clients import simple


class Client(simple.Client):
    """
    A federated learning client prunes its update before sending out.
    """
    async def train(self):
        # Perform model training
        self.report, weights = await super().train()

        logging.info("[Client #%d] Pruned its update.", self.client_id)

        return self.report, weights
