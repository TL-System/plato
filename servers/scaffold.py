"""
A federated learning server using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (https://arxiv.org/pdf/1910.06378.pdf)
"""
from collections import OrderedDict

from servers import FedAvgServer


class ScaffoldServer(FedAvgServer):
    """A federated learning server using the SCAFFOLD algorithm. """
    def __init__(self):
        super().__init__()
        self.c_server = None

    def federated_averaging(self, reports):
        """Aggregate weight updates from the clients and update c_server."""
        updated_weights = super().federated_averaging(reports)

        # Extracting updates from the reports
        updates = self.extract_client_updates(reports)

        # Extracting the delta_c
        update_cs = [report.delta_c
                     for report in reports]  # require a function

        # Update c_server
        if self.c_server == None:
            self.c_server = {
                name: self.trainer.zeros(weights.shape)
                for name, weights in updates[0].items()
            }

        for i, update_c in enumerate(update_cs):
            for name, delta_c in update_c.items():
                # Use weighted average by the number of samples
                self.c_server[name] += delta_c / Config().clients.per_round

        return updated_weights

    async def customize_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        server_response['c_server'] = self.c_server
        return server_response
