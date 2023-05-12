"""
A personalized federated learning client For APFL.
"""

from bases import personalized_client


class Client(personalized_client.Client):
    """A client to of APFL."""

    def inbound_received(self, inbound_processor):
        """Reloading the personalized model."""

        # always load the personalized model and the corresponding
        # ALPF's alpha for the subsequent learning
        loaded_status = self.load_personalized_model()

        self.trainer.extract_alpha(loaded_status)

        # assign the testset and testset sampler to the trainer
        self.trainer.set_testset(self.testset)
        self.trainer.set_testset_sampler(self.testset_sampler)
