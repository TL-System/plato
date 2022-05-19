"""
A basic federated learning client who sends weight updates to the server.
"""

from dataclasses import dataclass
import logging
import time

from plato.algorithms import registry as algorithms_registry
from plato.clients import base
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import registry as samplers_registry
from plato.trainers import registry as trainers_registry


@dataclass
class Report(base.Report):
    """Report from a simple client, to be sent to the federated learning server."""
    comm_time: float
    update_response: bool


class Client(base.Client):
    """A basic federated learning client who sends simple weight updates."""

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__()
        self.custom_model = model
        self.model = None

        self.datasource = datasource

        self.custom_algorithm = algorithm
        self.algorithm = None

        self.custom_trainer = trainer
        self.trainer = None

        self.trainset = None  # Training dataset
        self.testset = None  # Testing dataset
        self.sampler = None
        self.testset_sampler = None  # Sampler for the test set

        self.report = None

    def configure(self) -> None:
        """Prepare this client for training."""
        super().configure()
        if self.custom_model is not None:
            self.model = self.custom_model()
            self.custom_model = None

        if self.trainer is None and self.custom_trainer is None:
            self.trainer = trainers_registry.get(model=self.model)
        elif self.trainer is None and self.custom_trainer is not None:
            self.trainer = self.custom_trainer(model=self.model)
            self.custom_trainer = None

        self.trainer.set_client_id(self.client_id)

        if self.algorithm is None and self.custom_algorithm is None:
            self.algorithm = algorithms_registry.get(trainer=self.trainer)
        elif self.algorithm is None and self.custom_algorithm is not None:
            self.algorithm = self.custom_algorithm(trainer=self.trainer)
            self.custom_algorithm = None

        self.algorithm.set_client_id(self.client_id)

        # Pass inbound and outbound data payloads through processors for
        # additional data processing
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Client", client_id=self.client_id, trainer=self.trainer)

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        logging.info("[%s] Loading its data source...", self)

        if self.datasource is None or (hasattr(Config().data, 'reload_data')
                                       and Config().data.reload_data):
            self.datasource = datasources_registry.get(
                client_id=self.client_id)

        logging.info("[%s] Dataset size: %s", self,
                     self.datasource.num_train_examples())

        # Setting up the data sampler
        self.sampler = samplers_registry.get(self.datasource, self.client_id)

        if hasattr(Config().trainer, 'use_mindspore'):
            # MindSpore requires samplers to be used while constructing
            # the dataset
            self.trainset = self.datasource.get_train_set(self.sampler)
        else:
            # PyTorch uses samplers when loading data with a data loader
            self.trainset = self.datasource.get_train_set()

        if hasattr(Config().clients, 'do_test') and Config().clients.do_test:
            # Set the testset if local testing is needed
            self.testset = self.datasource.get_test_set()
            if hasattr(Config().data, 'testset_sampler'):
                # Set the sampler for test set
                self.testset_sampler = samplers_registry.get(self.datasource,
                                                             self.client_id,
                                                             testing=True)

    def load_payload(self, server_payload) -> None:
        """Loading the server model onto this client."""
        self.algorithm.load_weights(server_payload)

    async def train(self):
        """The machine learning training workload on a client."""
        logging.info("[%s] Started training in communication round #%s.", self,
                     self.current_round)

        # Perform model training
        try:
            training_time = self.trainer.train(self.trainset, self.sampler)
        except ValueError:
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if (hasattr(Config().clients, 'do_test') and Config().clients.do_test
            ) and (not hasattr(Config().clients, 'test_interval') or
                   self.current_round % Config().clients.test_interval == 0):
            accuracy = self.trainer.test(self.testset, self.testset_sampler)

            if accuracy == -1:
                # The testing process failed, disconnect from the server
                await self.sio.disconnect()

            if hasattr(Config().trainer, 'target_perplexity'):
                logging.info("[%s] Test perplexity: %.2f", self, accuracy)
            else:
                logging.info("[%s] Test accuracy: %.2f%%", self,
                             100 * accuracy)
        else:
            accuracy = 0

        comm_time = time.time()

        if hasattr(Config().clients,
                   'sleep_simulation') and Config().clients.sleep_simulation:
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]
            avg_training_time = Config().clients.avg_training_time
            self.report = Report(self.sampler.trainset_size(), accuracy,
                                 (avg_training_time + sleep_seconds) *
                                 Config().trainer.epochs, comm_time, False)
        else:
            self.report = Report(self.sampler.trainset_size(), accuracy,
                                 training_time, comm_time, False)

        return self.report, weights

    async def obtain_model_update(self, wall_time):
        """Retrieving a model update corresponding to a particular wall clock time."""
        model = self.trainer.obtain_model_update(wall_time)
        weights = self.algorithm.extract_weights(model)
        self.report.comm_time = time.time()
        self.report.update_response = True

        return self.report, weights

    def save_model(self, model_checkpoint):
        """ Saving the model to a model checkpoint. """
        self.trainer.save_model(model_checkpoint)

    def load_model(self, model_checkpoint):
        """ Loading the model from a model checkpoint. """
        self.trainer.load_model(model_checkpoint)
