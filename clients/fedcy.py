"""
A federated learning client of fedcy.
"""

from config import Config
from clients import SimpleClient


class CYClient(SimpleClient):
    """A federated learning client of fedcy."""
    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if 'local_epoch_num' in server_response:
            # Update the number of local epochs
            local_epoch_num = server_response['local_epoch_num']
            Config().trainer = Config().trainer._replace(
                epochs=local_epoch_num)
