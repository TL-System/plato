"""
Customized Client for PerFedRLNAS.
"""
from plato.clients import simple


class Client(simple.Client):
    """A FedRLNAS client. Different clients hold different models."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

    def process_server_response(self, server_response) -> None:
        subnet_config = server_response["subnet_config"]
        self.algorithm.model = self.algorithm.generate_client_model(subnet_config)
        self.trainer.model = self.algorithm.model
