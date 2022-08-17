"""
A basic federated learning client who sends weight updates to the server.
"""

import logging
import time
from types import SimpleNamespace

from plato.algorithms import registry as algorithms_registry
from plato.clients import base
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import registry as samplers_registry
from plato.trainers import registry as trainers_registry
from plato.utils import fonts


class Client(base.Client):
    """A basic federated learning client who sends simple weight updates."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__()
        self.custom_model = model
        self.model = None

        self.custom_datasource = datasource
        self.datasource = None

        self.custom_algorithm = algorithm
        self.algorithm = None

        self.custom_trainer = trainer
        self.trainer = None

        self.trainset = None  # Training dataset
        self.testset = None  # Testing dataset
        self.sampler = None
        self.testset_sampler = None  # Sampler for the test set

        self._report = None

    def configure(self) -> None:
        """Prepares this client for training."""
        super().configure()

        if self.model is None and self.custom_model is not None:
            self.model = self.custom_model

        if self.trainer is None and self.custom_trainer is None:
            self.trainer = trainers_registry.get(model=self.model)
        elif self.trainer is None and self.custom_trainer is not None:
            self.trainer = self.custom_trainer(model=self.model)

        self.trainer.set_client_id(self.client_id)

        if self.algorithm is None and self.custom_algorithm is None:
            self.algorithm = algorithms_registry.get(trainer=self.trainer)
        elif self.algorithm is None and self.custom_algorithm is not None:
            self.algorithm = self.custom_algorithm(trainer=self.trainer)

        self.algorithm.set_client_id(self.client_id)

        # Pass inbound and outbound data payloads through processors for
        # additional data processing
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Client", client_id=self.client_id, trainer=self.trainer
        )

    def load_data(self) -> None:
        """Generates data and loads them onto this client."""
        logging.info("[%s] Loading its data source...", self)

        if (
            self.datasource is None
            and self.custom_datasource is None
            or (hasattr(Config().data, "reload_data") and Config().data.reload_data)
        ):
            # The only case where Config().data.reload_data is set to true is
            # when clients with different client IDs need to load from different datasets,
            # such as in the pre-partitioned Federated EMNIST dataset. We do not support
            # reloading data from a custom datasource at this time.
            self.datasource = datasources_registry.get(client_id=self.client_id)
        elif self.datasource is None and self.custom_datasource is not None:
            self.datasource = self.custom_datasource()

        logging.info(
            "[%s] Dataset size: %s", self, self.datasource.num_train_examples()
        )

        # Setting up the data sampler
        self.sampler = samplers_registry.get(self.datasource, self.client_id)

        if hasattr(Config().trainer, "use_mindspore"):
            # MindSpore requires samplers to be used while constructing
            # the dataset
            self.trainset = self.datasource.get_train_set(self.sampler)
        else:
            # PyTorch uses samplers when loading data with a data loader
            self.trainset = self.datasource.get_train_set()

        if hasattr(Config().clients, "do_test") and Config().clients.do_test:
            # Set the testset if local testing is needed
            self.testset = self.datasource.get_test_set()
            if hasattr(Config().data, "testset_sampler"):
                # Set the sampler for test set
                self.testset_sampler = samplers_registry.get(
                    self.datasource, self.client_id, testing=True
                )

    def load_payload(self, server_payload) -> None:
        """Loads the server model onto this client."""
        self.algorithm.load_weights(server_payload)

    async def train(self):
        """The machine learning training workload on a client."""
        logging.info(
            fonts.colourize(
                f"[{self}] Started training in communication round #{self.current_round}."
            )
        )

        # Perform model training
        try:
            if hasattr(self.trainer, "current_round"):
                self.trainer.current_round = self.current_round
            training_time = self.trainer.train(self.trainset, self.sampler)
        except ValueError as exc:
            logging.info(
                fonts.colourize(f"[{self}] Error occurred during training: {exc}")
            )
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if (hasattr(Config().clients, "do_test") and Config().clients.do_test) and (
            not hasattr(Config().clients, "test_interval")
            or self.current_round % Config().clients.test_interval == 0
        ):
            accuracy = self.trainer.test(self.testset, self.testset_sampler)

            if accuracy == -1:
                # The testing process failed, disconnect from the server
                await self.sio.disconnect()

            if hasattr(Config().trainer, "target_perplexity"):
                logging.info("[%s] Test perplexity: %.2f", self, accuracy)
            else:
                logging.info("[%s] Test accuracy: %.2f%%", self, 100 * accuracy)
        else:
            accuracy = 0

        comm_time = time.time()

        if (
            hasattr(Config().clients, "sleep_simulation")
            and Config().clients.sleep_simulation
        ):
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]
            avg_training_time = Config().clients.avg_training_time

            report = SimpleNamespace(
                num_samples=self.sampler.num_samples(),
                accuracy=accuracy,
                training_time=(avg_training_time + sleep_seconds)
                * Config().trainer.epochs,
                comm_time=comm_time,
                update_response=False,
            )
        else:
            report = SimpleNamespace(
                num_samples=self.sampler.num_samples(),
                accuracy=accuracy,
                training_time=training_time,
                comm_time=comm_time,
                update_response=False,
            )

        self._report = self.customize_report(report)

        return self._report, weights

    async def obtain_model_update(self, wall_time):
        """Retrieves a model update corresponding to a particular wall clock time."""
        model = self.trainer.obtain_model_update(wall_time)
        weights = self.algorithm.extract_weights(model)
        self._report.comm_time = time.time()
        self._report.update_response = True

        return self._report, weights

    def save_model(self, model_checkpoint):
        """Saves the model to a model checkpoint."""
        self.trainer.save_model(model_checkpoint)

    def load_model(self, model_checkpoint):
        """Loads the model from a model checkpoint."""
        self.trainer.load_model(model_checkpoint)

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Customizes the report with any additional information."""
        return report
