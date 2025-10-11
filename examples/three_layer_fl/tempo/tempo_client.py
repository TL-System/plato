"""
A federated learning client of Tempo.
"""

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A federated learning client of Tempo."""

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        # Update the number of local epochs
        local_epoch_num = server_response["local_epoch_num"]
        Config().trainer = Config().trainer._replace(epochs=local_epoch_num)
