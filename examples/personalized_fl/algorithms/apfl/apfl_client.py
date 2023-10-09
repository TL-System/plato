"""
A personalized federated learning client For APFL.
"""

from pflbases import personalized_client


class Client(personalized_client.Client):
    """A client to of APFL."""

    def inbound_received(self, inbound_processor):
        """Reloading the personalized model."""
        super().inbound_received(inbound_processor)

        # loading previously saved ALPF's alpha for
        # the subsequent learning
        self.trainer.extract_alpha()
