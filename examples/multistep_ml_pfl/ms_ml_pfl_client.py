"""
A personalized federated learning client based on the multi-step meta-learning.
"""
import logging
import pickle
import sys
import time
from dataclasses import dataclass

from plato.datasources import registry as datasources_registry
from plato.samplers import registry as samplers_registry

from plato.config import Config
from plato.clients import simple


@dataclass
class Report(simple.Report):
    """A client report."""


class Client(simple.Client):
    """A federated learning client based on one-step meta-learning."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model=model,
                         datasource=datasource,
                         algorithm=algorithm,
                         trainer=trainer)
        # whether to perform the personalization test in each client
        self.do_meta_personalization_test = False
        # whether to let each client performs the local training based on its own data
        self.do_local_personalization = False

    # we rewrite the load_data function because each client must have
    #   the test data locally
    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        data_loading_start_time = time.perf_counter()
        logging.info("[Client #%d] Loading its data source...", self.client_id)

        if self.datasource is None:
            self.datasource = datasources_registry.get(
                client_id=self.client_id)

        self.data_loaded = True

        logging.info("[Client #%d] Dataset size: %s", self.client_id,
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

        # each client must have the test dataset
        self.testset = self.datasource.get_test_set()

        self.data_loading_time = time.perf_counter() - data_loading_start_time

    # corresponding to the customize_server_response in the server side
    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if 'meta_personalization_test' in server_response:
            self.do_meta_personalization_test = True

        elif 'local_personalization' in server_response:
            self.do_local_personalization = True

    async def payload_done(self, client_id, object_key) -> None:
        """ Upon receiving all the new payload from the server. """
        payload_size = 0

        if object_key is None:
            if isinstance(self.server_payload, list):
                for _data in self.server_payload:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            elif isinstance(self.server_payload, dict):
                for key, value in self.server_payload.items():
                    payload_size += sys.getsizeof(pickle.dumps({key: value}))
            else:
                payload_size = sys.getsizeof(pickle.dumps(self.server_payload))
        else:
            self.server_payload = self.s3_client.receive_from_s3(object_key)
            payload_size = sys.getsizeof(pickle.dumps(self.server_payload))

        assert client_id == self.client_id

        logging.info(
            "[Client #%d] Received %s MB of payload data from the server.",
            client_id, round(payload_size / 1024**2, 2))

        self.load_payload(self.server_payload)
        self.server_payload = None

        if self.do_meta_personalization_test:
            # Train a personalized model based on the current meta model and test it
            # This report only contains accuracy of its personalized model
            report = await self.perform_meta_personalization()
            payload = 'meta_personalization_accuracy'
            self.do_meta_personalization_test = False
        elif self.do_local_personalization:
            report = await self.perform_local_personalization()
            payload = 'local_personalization_accuracy'
            self.do_local_personalization = False
        else:
            # Regular local training of FL
            report, payload = await self.train()
            logging.info("[Client #%d] Model trained.", client_id)

        # Sending the client report as metadata to the server (payload to follow)
        await self.sio.emit('client_report', {'report': pickle.dumps(report)})

        # Sending the client training payload to the server
        await self.send(payload)

    async def perform_meta_personalization(self):
        """A client performs the personalization by first updating the
            received meta-model with one-step of SGD and then testing
            the fine-tuned model based on the rest of data.
            This function actually belongs to the test part as the meta-model is
            needed to be updated on the test set.
        """
        logging.info(
            "[Client #%d] Started training a personalized model by updating the meta model.",
            self.client_id)

        # Train a personalized model and test it
        self.trainer.test_meta_personalization = True
        personalization_accuracy = self.trainer.test(self.testset,
                                                     self.sampler)
        self.trainer.test_meta_personalization = False

        if personalization_accuracy == 0:
            # The testing process failed, disconnect from the server
            await self.sio.disconnect()

        logging.info(
            "[Client #{:d}] meta Personlization accuracy: {:.2f}%".format(
                self.client_id, personalization_accuracy))

        return personalization_accuracy

    async def perform_local_personalization(self):
        """ A client performs the local personalization based on its own
            local data. The model is trained directly from the initialization
            parameters within severl epoches.
            we assign this local personalization to the test phase.
        """
        logging.info(
            "[Client #%d] Started training a personalized model direclty \
            based on its local trainset.", self.client_id)

        local_personalization_accuracy = self.trainer.perform_local_personalization_test(
            config=Config().trainer._asdict(),
            trainset=self.trainset,
            testset=self.testset,
            sampler=self.sampler)
        if local_personalization_accuracy == 0:
            # The testing process failed, disconnect from the server
            await self.sio.disconnect()

        logging.info(
            "[Client #{:d}] Local personlization accuracy: {:.2f}%".format(
                self.client_id, local_personalization_accuracy))

        return local_personalization_accuracy
