"""
Implementation of the FedEMA's clients

"""

from plato.clients import ssl_simple as ssl_client
from plato.config import Config


class Client(ssl_client.Client):
    """A personalized federated learning client with self-supervised support
        for the FedEMA method."""

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

        # the genralization scale of this client computed on
        # the server side
        self.generality_scale = 0.0

        # the moving average weight used to
        # fusion the local and global model
        # set to 1.0 for only maintaining the previous model
        # by rejecting the shared global model
        self.local_global_ema_genrlz_scale = 1.0

    def perform_local_global_moving_average(self, server_payload):
        """ Perform the local global moving average when necessary. """

        # computed the moving average weight based on the
        # generalization scale sent from the server
        self.local_global_ema_genrlz_scale = min(self.generality_scale, 1)

        # update the model with moving average.
        self.algorithm.load_weights_moving_average(
            server_payload, average_scale=self.local_global_ema_genrlz_scale)

    def load_payload(self, server_payload) -> None:
        """Loading the server model onto this client
            using the method of moving average. """
        #self.algorithm.load_weights(server_payload)
        # the received server_payload is a ordered dict containing the
        # parameter name and the parameter data as:
        #   {para_name: para_data}
        self.perform_local_global_moving_average(server_payload)

    def process_server_response(self, server_response) -> None:
        """Additional client-specific processing on the server response."""
        # the server response should contain the generalization scale computed
        # for this client
        self.generality_scale = server_response[
            "generalization_divergence_scale"]
