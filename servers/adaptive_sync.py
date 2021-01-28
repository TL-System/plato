"""
A simple federated learning server using federated averaging.
"""
from servers import FedAvgServer


class AdaptiveSyncServer(FedAvgServer):
    """Federated averaging server with Adaptive Synchronization Frequency."""
    def __init__(self):
        super().__init__()
        self.previous_model = None

    async def customize_server_response(self, server_response):
        """Customizing the server response with any additional information."""
        server_response['sync_frequency'] = self.algorithm.sync_frequency
        return server_response
