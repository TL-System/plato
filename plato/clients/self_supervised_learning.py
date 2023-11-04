"""
A self-supervised learning (SSL) client prepares a personalized datasource for
the personalization process, which will be performed after finishing the FL
training process with SSL. 

Specifically, the conventional FL training process with SSL will train the model
with the datasource and objective function of SSL. Yet, the datasource used in
personalization should be one of supervised learning. Therefore, a client needs
to prepare the personalized datasource. 
"""

from plato.datasources import registry as datasources_registry
from plato.clients import simple


class Client(simple.Client):
    """An SSL client to prepare the datasource for personalization."""

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
        # The datasource used in personalization
        self.personalized_datasource = None

    def configure(self) -> None:
        """Prepare this client for training."""
        super().configure()

        # Get the personalized datasource
        if self.personalized_datasource is None:
            personalized_datasource = datasources_registry.get()

        # Set the train and the test set for the trainer
        self.trainer.set_personalized_datasets(
            personalized_datasource.get_train_set(),
            personalized_datasource.get_test_set(),
        )
