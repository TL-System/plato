"""
The federated learning server for Adaptive Synchronization Frequency.

Reference:

C. Chen, et al. "GIFT: Towards Accurate and Efficient Federated
Learning withGradient-Instructed Frequency Tuning," found in docs/papers.
"""
from servers import FedAvgServer


class AdaptiveSyncServer(FedAvgServer):
    """Federated averaging server with Adaptive Synchronization Frequency."""
    def __init__(self):
        super().__init__()
        self.previous_model = None

    async def customize_server_response(self, server_response):
        """Customizing the server response with any additional information."""
        server_response['sync_frequency'] = self.trainer.sync_frequency
        return server_response

    @staticmethod
    def is_valid_server_type(server_type):
        """Determine if the server type is valid. """
        return server_type == 'adaptive_sync'

    @staticmethod
    def get_server():
        """Returns an instance of this server. """
        return AdaptiveSyncServer()