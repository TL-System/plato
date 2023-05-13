"""
A personalized federated learning client performing 
self-supervised learning (SSL).

"""

import logging

from plato.samplers import registry as samplers_registry
from plato.config import Config

from bases import personalized_client


class Client(personalized_client.Client):
    """A basic personalized federated learning client for self-supervised learning."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        personalized_model=None,
        personalized_datasource=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            personalized_model=personalized_model,
        )

        # the personalized datasource
        # By default, if `personalized_datasource` is not set up, it will
        # be equal to the `datasource`
        self.custom_personalized_datasource = personalized_datasource
        self.personalized_datasource = None

        # dataset for personalization
        self.personalized_trainset = None
        self.personalized_testset = None

        # By default, if `personalized_sampler` is not set up, it will
        # be equal to the `sampler`.
        self.personalized_sampler = None
        self.personalized_testset_sampler = None

    def configure(self) -> None:
        """Prepares this client for training."""
        super().configure()

        if self.personalized_datasource is None:
            logging.info("[%s] Define its personalized Dataset", self)
            self.personalized_datasource = self.custom_personalized_datasource(
                transform_block=Config().algorithm.personalization.data_transforms._asdict()
            )

        # Setting up the data sampler for personalization
        sampler_type = (
            Config().algorithm.personalization.sampler
            if hasattr(Config().algorithm.personalization, "sampler")
            else Config().data.sampler
        )
        self.personalized_sampler = samplers_registry.get(
            self.personalized_datasource,
            self.client_id,
            sampler_type=sampler_type,
        )

        sampler_type = (
            Config().algorithm.personalization.testset_sampler
            if hasattr(Config().algorithm.personalization, "testset_sampler")
            else Config().algorithm.personalization.testset_sampler
        )
        # Set the sampler for test set
        self.personalized_testset_sampler = samplers_registry.get(
            self.personalized_datasource,
            self.client_id,
            testing=True,
            sampler_type=sampler_type,
        )

        # obtain the train/test set for personalization
        self.personalized_trainset = self.personalized_datasource.get_train_set()
        self.personalized_testset = self.personalized_datasource.get_test_set()

        # set personalized terms for the trainer
        self.trainer.set_personalized_trainset(self.personalized_trainset)
        self.trainer.set_personalized_trainset_sampler(self.personalized_sampler)
        self.trainer.set_personalized_testset(self.personalized_testset)
        self.trainer.set_personalized_testset_sampler(self.personalized_testset_sampler)
