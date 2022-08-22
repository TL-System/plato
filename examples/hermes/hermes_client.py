"""
A federated learning client using Hermes
"""

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A Hermes federated learning client."""

    def process_server_response(self, server_response):
        """Initialize the path for the extra client payload - mask"""
        self.trainer.extra_payload_path = (
            f"{Config().params['checkpoint_path']}/{Config().trainer.model_name}"
            f"_client{self.client_id}_mask.pth"
        )
