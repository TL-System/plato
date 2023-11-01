"""
A personalized federated learning client performing 
self-supervised learning (SSL).

"""

import logging

from plato.samplers import registry as samplers_registry
from plato.config import Config

from plato.clients import simple


class Client(simple.Client):
    """A basic personalized federated learning client for self-supervised learning."""

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

        # dataset for personalization
        self.personalized_trainset = None
        self.personalized_testset = None

    def configure(self) -> None:
        """Prepares this client for training."""
        super().configure()

        if self.personalized_datasource is None:
            transforms_block = (
                {}
                if not hasattr(Config().algorithm.personalization, "data_transforms")
                else Config().algorithm.personalization.data_transforms._asdict()
            )
            logging.info("Defining the personalized datasource:")
            self.personalized_datasource = self.custom_personalized_datasource(
                transforms_block=transforms_block
            )

        # obtain the train/test set for personalization
        self.personalized_trainset = self.personalized_datasource.get_train_set()
        self.personalized_testset = self.personalized_datasource.get_test_set()

        # set personalized terms for the trainer
        self.trainer.set_personalized_trainset(self.personalized_trainset)
        self.trainer.set_personalized_testset(self.personalized_testset)
