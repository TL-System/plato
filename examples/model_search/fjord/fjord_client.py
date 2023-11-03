"""
Customized Client for FjORD.
"""
from plato.config import Config
from plato.clients import simple


class Client(simple.Client):
    """A federated learning server using the FjORD algorithm."""

    def process_server_response(self, server_response) -> None:
        rate = server_response["rate"]
        self.trainer.rate = rate
        self.algorithm.model = self.model(
            model_rate=rate, **Config().parameters.client_model._asdict()
        )
        self.trainer.model = self.algorithm.model
