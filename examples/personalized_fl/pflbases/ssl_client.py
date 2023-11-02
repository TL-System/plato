"""
A base client for self-supervised learning (SSL).
"""

from plato.datasources import registry as datasources_registry
from plato.clients import simple


class Client(simple.Client):
    """A base client to prepare the datasource for self-supervised learning
    in the personalization."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=trainer_callbacks,
        )
        self.personalized_datasource = None

    def configure(self) -> None:
        """Prepares this client for training."""
        super().configure()

        # Get the personalized datasource
        if self.personalized_datasource is None:
            personalized_datasource = datasources_registry.get()

        # Set the train and the test set for the trainer
        self.trainer.set_personalized_datasets(
            personalized_datasource.get_train_set(),
            personalized_datasource.get_test_set(),
        )
