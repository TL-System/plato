"""
A base client for self-supervised learning (SSL).
"""

from plato.datasources import registry as datasources_registry
from plato.clients import simple


class Client(simple.Client):
    """A base to prepare the datasource for self-supervised learning
    in the personalization."""

    def configure(self) -> None:
        """Prepares this client for training."""
        super().configure()

        # Get the personalized datasource
        personalized_datasource = datasources_registry.get()

        # Set the train and the test set for the trainer
        self.trainer.set_personalized_trainset(personalized_datasource.get_train_set())
        self.trainer.set_personalized_testset(personalized_datasource.get_test_set())
