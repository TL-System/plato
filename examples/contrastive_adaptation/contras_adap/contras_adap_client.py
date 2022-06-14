"""
Implementation of the FedEMA's clients

"""

from plato.clients import ssl_simple as ssl_client


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
        self.local_global_ema_scale = 1.0

    def load_payload(self, server_payload) -> None:
        """Loading the server model onto this client
            using the method of moving average. """
        #self.algorithm.load_weights(server_payload)

        # computed the moving average weight
        self.local_global_ema_scale = min(self.generality_scale, 1)

        # update the model with moving average.
        self.algorithm.load_weights_moving_average(
            server_payload, average_scale=self.local_global_ema_scale)

    def process_server_response(self, server_response) -> None:
        """Additional client-specific processing on the server response."""
        # the server response should contain the generalization scale computed
        # for this client
        self.generality_scale = server_response[
            "generalization_divergence_scale"]
