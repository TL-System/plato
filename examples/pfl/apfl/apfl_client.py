"""
A personalized federated learning client For APFL.
"""

from bases import personalized_client


class Client(personalized_client.Client):
    """A client to of APFL."""

    def inbound_received(self, inbound_processor):
        """Reloading the personalized model."""
        super().inbound_received(inbound_processor)

        # always load the personalized model and the corresponding
        # ALPF's alpha for the subsequent learning
        loaded_status = self.load_personalized_model()

        self.trainer.extract_alpha(loaded_status)
