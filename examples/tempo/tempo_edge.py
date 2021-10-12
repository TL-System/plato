"""
A federated learning client at edge server of Tempo.
"""

from plato.config import Config

from plato.clients import edge


class Client(edge.Client):
    """A federated learning edge client of Tempo."""
    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        super().process_server_response(server_response)

        if 'local_epoch_num' in server_response:
            local_epoch_list = server_response['local_epoch_num']
            if hasattr(Config().clients,
                       'simulation') and Config().clients.simulation:
                index = self.client_id - Config().clients.per_round - 1
            else:
                index = self.client_id - Config().clients.total_clients - 1

            local_epoch_num = local_epoch_list[index]
            # Update the number of local epochs
            Config().trainer = Config().trainer._replace(
                epochs=local_epoch_num)
