"""
A personalized federated learning client For Ditto.
"""

from pflbases import personalized_client


class Client(personalized_client.Client):
    """A client to of Ditto."""

    def inbound_received(self, inbound_processor):
        """Reloading the personalized model."""

        # always load the personalized model
        # for the subsequent jointly training
        self.load_personalized_model()
        # assign the testset and testset sampler to the trainer
        self.trainer.set_testset(self.testset)
        self.trainer.set_testset_sampler(self.testset_sampler)
