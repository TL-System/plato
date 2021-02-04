"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" 
(https://arxiv.org/pdf/1910.06378.pdf)
"""

from servers import FedAvgServer
from config import Config
import logging


class ScaffoldServer(FedAvgServer):
    """A federated learning server using the SCAFFOLD algorithm. """
    def __init__(self):
        super().__init__()
        self.c_server = None

    def federated_averaging(self, reports):
        """Aggregate weight updates and delta_c updates from the clients."""
        updated_weights = super().federated_averaging(reports)
        logging.info('Server received updated_weights.')

        # Extracting delta_c
        update_cs = [report.delta_c for report in reports]
        logging.info('Server received updated_delta_c from clients')

        # initilize c_server
        if self.c_server is None:
            self.c_server = [0] * len(update_cs[0])

        # Update c_server
        for update_c in update_cs:
            for j, delta in enumerate(update_c):
                self.c_server[j] += delta / Config().clients.total_clients

        return updated_weights

    async def customize_server_response(self, server_response):
        """Add 'c_server' into the server response."""
        server_response['c_server'] = True
        return server_response
