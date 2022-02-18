"""
A cross-silo federated learning server using FedSaw, as either edge or central servers.
"""

from plato.servers import fedavg_cs


class Server(fedavg_cs.Server):
    """Cross-silo federated learning server using FedSaw."""
    def extract_client_updates(self, updates):
        """ Extract the model weight updates from client updates. """
        updates_received = [payload for (__, payload, __) in updates]
        return updates_received
