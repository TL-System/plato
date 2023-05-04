"""
A basic personalized federated learning client who performs the 
self-supervised learning.

"""

import logging
import time
from types import SimpleNamespace

from examples.pfl.bases import simple_personalized
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.datasources import transform_registry
from plato.samplers import registry as samplers_registry
from plato.utils import fonts


class Client(simple_personalized.Client):
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

        # the data transformer for personalized data
        # By default, if transform is not set up, it will utilize the
        # default transform defined within Plato's data source.
        self.train_transform = None
        self.test_transform = None
        self.data_transforms = {}
        self.personalized_train_transform = None
        self.personalized_test_transform = None
        self.personalized_data_transforms = {}

    def get_personalized_model_params(self):
        """Get the params of the personalized model."""
        pers_model_params = Config().parameters.personalized_model._asdict()
        pers_model_params["input_dim"] = self.trainer.model.encoding_dim
        pers_model_params["output_dim"] = pers_model_params["num_classes"]
        return pers_model_params

    def define_data_transform(self):
        """Define the data transform for normal personalized federated learning."""
        if hasattr(Config().data, "train_transform"):
            self.train_transform = transform_registry.get()
            logging.info("[%s] Defined train transform %s", self, self.train_transform)

        if hasattr(Config().data, "test_transform"):
            transform_name = Config().data.test_transform
            transform_params = (
                Config().parameters.test_transform._asdict()
                if hasattr(Config().parameters, "test_transform")
                else {}
            )
            self.test_transform = transform_registry.get(
                data_transform_name=transform_name,
                data_transform_params=transform_params,
            )
            logging.info("[%s] Defined test transform %s", self, self.test_transform)

        if self.train_transform is not None:
            self.data_transforms.update({"train_transform": self.train_transform})
        if self.test_transform is not None:
            self.data_transforms.update({"test_transform": self.test_transform})

    def define_personalized_data_transform(self):
        """Define the data transform for personalized personalized federated learning."""
        if hasattr(Config().data, "personalized_train_transform"):
            transform_name = Config().data.personalized_train_transform
            transform_params = (
                Config().parameters.personalized_train_transform._asdict()
                if hasattr(Config().parameters, "personalized_train_transform")
                else {}
            )
            self.personalized_train_transform = transform_registry.get(
                data_transform_name=transform_name,
                data_transform_params=transform_params,
            )
            logging.info(
                "[%s] Defined personalized train transform %s",
                self,
                self.personalized_train_transform,
            )

        if hasattr(Config().data, "personalized_test_transform"):
            transform_name = Config().data.personalized_test_transform
            transform_params = (
                Config().parameters.personalized_test_transform._asdict()
                if hasattr(Config().parameters, "personalized_test_transform")
                else {}
            )
            self.personalized_test_transform = transform_registry.get(
                data_transform_name=transform_name,
                data_transform_params=transform_params,
            )
            logging.info(
                "[%s] Defined personalized test transform %s",
                self,
                self.personalized_test_transform,
            )

        if self.personalized_train_transform is not None:
            self.personalized_data_transforms.update(
                {"train_transform": self.personalized_train_transform}
            )
        if self.personalized_test_transform is not None:
            self.personalized_data_transforms.update(
                {"test_transform": self.personalized_test_transform}
            )

    def _load_data(self) -> None:
        """Generates data and loads them onto this client."""

        # The only case where Config().data.reload_data is set to true is
        # when clients with different client IDs need to load from different datasets,
        # such as in the pre-partitioned Federated EMNIST dataset. We do not support
        # reloading data from a custom datasource at this time.
        if (
            self.datasource is None
            or hasattr(Config().data, "reload_data")
            and Config().data.reload_data
        ):
            logging.info("[%s] Loading its data source...", self)

            self.define_data_transform()

            if self.custom_datasource is None:

                self.datasource = datasources_registry.get(
                    client_id=self.client_id, **self.data_transforms
                )
            elif self.custom_datasource is not None:
                self.datasource = self.custom_datasource(**self.data_transforms)

            logging.info(
                "[%s] Dataset size: %s; Transform: %s",
                self,
                self.datasource.num_train_examples(),
                self.data_transforms,
            )

        if self.personalized_datasource is None:
            personalized_datasource = (
                Config().data.personalized_datasource
                if hasattr(Config().data, "personalized_datasource")
                else Config().data.datasource
            )
            logging.info(
                "[%s] Loading its personalized data source %s",
                self,
                personalized_datasource,
            )
            self.define_personalized_data_transform()

            if self.custom_personalized_datasource is None:
                self.personalized_datasource = datasources_registry.get(
                    client_id=self.client_id, **self.personalized_data_transforms
                )
            elif self.custom_personalized_datasource is not None:
                self.personalized_datasource = self.custom_personalized_datasource(
                    **self.personalized_data_transforms
                )

            logging.info(
                "[%s] Personalized Dataset size: %s; Transforme: %s",
                self,
                self.personalized_datasource.num_train_examples(),
                self.personalized_data_transforms,
            )

    def configure(self) -> None:
        """Prepares this client for training."""
        super().configure()

        # Setting up the data sampler for personalization
        if self.personalized_datasource:
            sampler_type = (
                Config().data.personalized_sampler
                if hasattr(Config().data, "personalized_sampler")
                else Config().data.sampler
            )
            self.personalized_sampler = samplers_registry.get(
                self.personalized_datasource,
                self.client_id,
                sampler_type=sampler_type,
            )

            if (
                hasattr(Config().clients, "do_personalized_test")
                and Config().clients.do_personalized_test
            ):
                sampler_type = (
                    Config().data.personalized_test_sampler
                    if hasattr(Config().data, "personalized_testset_sampler")
                    else Config().data.testset_sampler
                )
                # Set the sampler for test set
                self.personalized_testset_sampler = samplers_registry.get(
                    self.personalized_datasource,
                    self.client_id,
                    testing=True,
                    sampler_type=sampler_type,
                )

    def _allocate_data(self) -> None:
        """Allocate training or testing dataset of this client."""
        super()._allocate_data()
        if hasattr(Config().trainer, "use_mindspore"):
            # MindSpore requires samplers to be used while constructing
            # the dataset
            self.personalized_trainset = self.personalized_datasource.get_train_set(
                self.sampler
            )

        else:
            # PyTorch uses samplers when loading data with a data loader
            self.personalized_trainset = self.personalized_datasource.get_train_set()

        if (
            hasattr(Config().clients, "do_personalized_test")
            and Config().clients.do_personalized_test
        ):
            # Set the personalized testset if local testing is needed
            self.personalized_testset = self.personalized_datasource.get_test_set()


    async def _train(self):
        """The machine learning training workload on a client.

        To make it flexible, there are three training mode.

        """

        accuracy = -1
        training_time = 0.0

        if hasattr(self.trainer, "current_round"):
            self.trainer.current_round = self.current_round

        if self.is_personalized_learn():
            logging.info(
                fonts.colourize(
                    f"[{self}] Started personalized training in the communication round #{self.current_round}.",
                    colour="blue",
                )
            )
            try:
                training_time, accuracy = self.trainer.personalized_train(
                    self.personalized_trainset,
                    self.personalized_sampler,
                    testset=self.personalized_testset,
                    testset_sampler=self.personalized_testset_sampler,
                )
            except ValueError:
                await self.sio.disconnect()

        else:
            logging.info(
                fonts.colourize(
                    f"[{self}] Started training in communication round #{self.current_round}."
                )
            )
            try:
                training_time = self.trainer.train(self.trainset, self.sampler)
            except ValueError:
                await self.sio.disconnect()

            if (hasattr(Config().clients, "do_test") and Config().clients.do_test) and (
                hasattr(Config().clients, "test_interval")
                and self.current_round % Config().clients.test_interval == 0
            ):
                accuracy = self.trainer.test(self.testset, self.testset_sampler)
            else:
                accuracy = 0

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if accuracy == -1:
            # The testing process failed, disconnect from the server
            await self.sio.disconnect()

        # Do not print the accuracy if it is not computed
        if accuracy != 0:
            if hasattr(Config().trainer, "target_perplexity"):
                logging.info("[%s] Test perplexity: %.2f", self, accuracy)
            else:
                logging.info("[%s] Test accuracy: %.2f%%", self, 100 * accuracy)

        comm_time = time.time()

        if (
            hasattr(Config().clients, "sleep_simulation")
            and Config().clients.sleep_simulation
        ):
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]
            avg_training_time = Config().clients.avg_training_time

            training_time = (
                avg_training_time + sleep_seconds
            ) * Config().trainer.epochs

        report = SimpleNamespace(
            client_id=self.client_id,
            num_samples=self.sampler.num_samples(),
            accuracy=accuracy,
            training_time=training_time,
            comm_time=comm_time,
            update_response=False,
        )

        self._report = self.customize_report(report)

        return self._report, weights
