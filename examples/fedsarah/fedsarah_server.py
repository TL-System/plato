"""
A customized server for FedSarah.

Reference: Ngunyen et al., "SARAH: A Novel Method for Machine Learning Problems
Using Stochastic Recursive Gradient." (https://arxiv.org/pdf/1703.00102.pdf)

"""
from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedSarah algorithm."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.server_control_variates = None
        self.control_variates_received = None

    def weights_received(self, weights_received):
        """Extract the model weights and control variates from clients updates."""
        self.control_variates_received = [weight[1] for weight in weights_received]
        return [weight[0] for weight in weights_received]

    def weights_aggregated(self, updates):
        """Method called after the updated weights have been aggregated."""
        # Initialize server control variates
        self.server_control_variates = [0] * len(self.control_variates_received[0])

        # Update server control variates
        for control_variates in self.control_variates_received:
            for j, control_variate in enumerate(control_variates):
                self.server_control_variates[j] += (
                    control_variate / Config().clients.total_clients
                )

    def customize_server_payload(self, payload):
        "Add server control variates into the server payload."
        return [payload, self.server_control_variates]
