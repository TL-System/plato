"""
A personalized federated learning client for FedBABU.
"""


from bases import personalized_client


class Client(personalized_client.Client):
    """A FedBABU federated learning client."""

    def inbound_received(self, inbound_processor):
        """Reloading the personalized model for this client before any operations."""
        super().inbound_received(inbound_processor)

        # always loading the personalized model whose head will
        # be assigned to the received payload
        self.load_personalized_model()
