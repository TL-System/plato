"""
The client for paper system-heterogenous federated learning through architecture search.
"""
from plato.config import Config
from plato.clients import simple


class Client(simple.Client):
    """A federated learning server using the ElasticArch algorithm."""

    def process_server_response(self, server_response) -> None:
        config = server_response["config"]
        self.algorithm.model = self.model(
            configs=config, **Config().parameters.client_model._asdict()
        )
        self.trainer.model = self.algorithm.model
