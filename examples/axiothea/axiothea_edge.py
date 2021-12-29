"""
A federated learning client at the edge server in a cross-silo training workload.
"""

import logging

from plato.clients import edge


class Client(edge.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""
    async def train(self):
        logging.info("[Edge Server #%d] Training on an Axiothea edge server.",
                     self.client_id)

        # Perform model training
        report, weights = await super().train()
        return report, weights
