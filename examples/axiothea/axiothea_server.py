"""
A cross-silo federated learning server using Axiothea, as either edge or central servers.
"""

from plato.config import Config
from plato.servers import fedavg_cs


class Server(fedavg_cs.Server):
    """Cross-silo federated learning server using Axiothea."""
    def customize_server_payload(self, payload):
        """ Customize the server payload before sending to its clients. """
        if Config().is_edge_server():
            # might add noise to aggreagted models
            pass

        return payload
