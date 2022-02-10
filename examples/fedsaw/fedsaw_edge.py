"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import logging

from plato.clients import edge


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""
    async def train(self):
        # Perform model training
        self.report, weights = await super().train()

        logging.info("[Edge Server #%d] Pruned its aggregated updates.",
                     self.client_id)

        return self.report, weights
