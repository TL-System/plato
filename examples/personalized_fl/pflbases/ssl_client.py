"""
A personalized federated learning client performing 
self-supervised learning (SSL).

"""

from plato.datasources import registry as datasources_registry
from plato.clients import simple


class Client(simple.Client):
    """A basic personalized federated learning client for self-supervised learning."""

    def configure(self) -> None:
        """Prepares this client for training."""
        super().configure()

        personalized_datasource = datasources_registry.get()

        # set personalized terms for the trainer
        self.trainer.set_personalized_trainset(personalized_datasource.get_train_set())
        self.trainer.set_personalized_testset(personalized_datasource.get_test_set())
