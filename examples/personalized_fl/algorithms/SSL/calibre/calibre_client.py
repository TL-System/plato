"""
A client for Calibre algorithm.
"""

from pflbases import ssl_client


class Client(ssl_client.Client):
    """A basic personalized federated learning client for self-supervised learning."""

    def inbound_received(self, inbound_processor):
        """Setting personalized datasets and the samplers to the trainer."""
        super().inbound_received(inbound_processor)

        # always set personalized terms for the trainer
        self.trainer.set_personalized_trainset(self.personalized_trainset)
        self.trainer.set_personalized_trainset_sampler(self.personalized_sampler)
        self.trainer.set_personalized_testset(self.personalized_testset)
        self.trainer.set_personalized_testset_sampler(self.personalized_testset_sampler)
