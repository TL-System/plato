"""
A customized trainer for federated unlearning.

Reference: Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining." in Proc. INFOCOM, 2022 https://arxiv.org/abs/2203.07320

"""
import logging
import pickle
import sys

import unlearning_iid
from plato.clients import simple
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.samplers import registry as samplers_registry


class Client(simple.Client):
    """A federated learning client of federated unlearning."""

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.current_round = 0
        self.testset_sampler = None

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        logging.info("[%s] Loading its data source...", self)

        if self.datasource is None or (hasattr(Config().data, 'reload_data')
                                       and Config().data.reload_data):
            self.datasource = datasources_registry.get(
                client_id=self.client_id)

        self.data_loaded = True

        logging.info("[%s] Dataset size: %s", self,
                     self.datasource.num_train_examples())

        need_delete = self.current_round >= Config().clients.data_deleted_round
        # Setting up the data sampler
        self.sampler = unlearning_iid.Sampler(self.datasource, self.client_id,
                                           False, need_delete)

        if hasattr(Config().trainer, 'use_mindspore'):
            # MindSpore requires samplers to be used while constructing
            # the dataset
            self.trainset = self.datasource.get_train_set(self.sampler)
        else:
            # PyTorch uses samplers when loading data with a data loader
            self.trainset = self.datasource.get_train_set()

        if Config().clients.do_test:
            # Set the testset if local testing is needed
            self.testset = self.datasource.get_test_set()
            if hasattr(Config().data, 'testset_sampler'):
                # Set the sampler for test set
                self.testset_sampler = samplers_registry.get(self.datasource,
                                                             self.client_id,
                                                             testing=True)

    async def payload_to_arrive(self, response) -> None:
        """ Upon receiving a response from the server. """
        self.process_server_response(response)

        self.current_round = response["current_round"]

        # Update (virtual) client id for client, trainer and algorithm
        if hasattr(Config().clients,
                   'simulation') and Config().clients.simulation:
            self.client_id = response['id']
            self.configure()

        logging.info("[Client #%d] Selected by the server.", self.client_id)

        if self.current_round == Config().clients.data_deleted_round:
            self.data_loaded = False

        if (hasattr(Config().data, 'reload_data')
                and Config().data.reload_data) or not self.data_loaded:
            self.load_data()

        if hasattr(Config().clients,
                   'comm_simulation') and Config().clients.comm_simulation:
            payload_filename = response['payload_filename']
            with open(payload_filename, 'rb') as payload_file:
                self.server_payload = pickle.load(payload_file)

            payload_size = sys.getsizeof(pickle.dumps(self.server_payload))

            logging.info(
                "[%s] Received %.2f MB of payload data from the server (simulated).",
                self, payload_size / 1024**2)

            self.server_payload = self.inbound_processor.process(
                self.server_payload)

            await self.start_training()
