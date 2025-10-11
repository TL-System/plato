"""
A federated learning client at edge server of Tempo.
"""

from plato.clients import edge
from plato.config import Config


class Client(edge.Client):
    """A federated learning edge client of Tempo."""

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        super().process_server_response(server_response)

        local_epoch_list = server_response["local_epoch_num"]
        index = self.client_id - Config().clients.total_clients - 1

        try:
            local_epoch_num = local_epoch_list[index]
        except (IndexError, TypeError):
            # Fallback: use first element if list, or the value itself if scalar
            local_epoch_num = (
                local_epoch_list[0]
                if isinstance(local_epoch_list, list) and local_epoch_list
                else Config().trainer.epochs
            )

        # Update the number of local training epochs
        Config().trainer = Config().trainer._replace(epochs=local_epoch_num)
