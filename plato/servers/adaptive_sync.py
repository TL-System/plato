"""
The federated learning server for Adaptive Synchronization Frequency.

Reference:

C. Chen, et al. "GIFT: Towards Accurate and Efficient Federated
Learning withGradient-Instructed Frequency Tuning," found in docs/papers.
"""
from plato.servers import fedavg


class Server(fedavg.Server):
    """Federated averaging server with Adaptive Synchronization Frequency."""
    async def customize_server_response(self, server_response):
        """Customizing the server response with any additional information."""
        server_response['sync_frequency'] = self.algorithm.sync_frequency
        return server_response
