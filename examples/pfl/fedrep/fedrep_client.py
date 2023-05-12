"""
A personalized federated learning client for FedRep.
"""


from bases import personalized_client


class Client(personalized_client.Client):
    """A FedRep federated learning client."""

    def inbound_received(self, inbound_processor):
        """Reloading the personalized model for this client before any operations."""
        self.load_personalized_model()

        # assign the testset and testset sampler to the trainer
        self.trainer.set_testset(self.testset)
        self.trainer.set_testset_sampler(self.testset_sampler)
