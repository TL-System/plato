"""
The federated learning server for Adaptive Parameter Freezing.

Reference:

C. Chen, et al. "Communication-Efficient Federated Learning with Adaptive
Parameter Freezing," found in docs/papers.
"""
from servers import FedAvgServer


class AdaptiveFreezingServer(FedAvgServer):
    """Federated averaging server for Adaptive Parameter Freezing."""
    @staticmethod
    def is_valid_server_type(server_type):
        """Determine if the server type is valid. """
        return server_type == 'adaptive_freezing'

    @staticmethod
    def get_server():
        """Returns an instance of this server. """
        return AdaptiveFreezingServer()