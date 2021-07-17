"""
A split learning server.
"""

import logging
import os
import pickle
import sys
import time
from copy import deepcopy
from itertools import chain

import torch
from plato.config import Config
from plato.samplers import all_inclusive
from plato.servers import fedavg


class Server(fedavg.Server):
    """The split learning server."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.select_client_socket = None
        self.selected_client_id = None
        self.last_selected_client_id = None

    def choose_clients(self):
        assert self.clients_per_round == 1
        if self.last_selected_client_id is None:
            # 1st train loop
            self.last_selected_client_id = -1  # Skip this snippet
            self.selected_client_id = 1
        else:
            self.last_selected_client_id = self.selected_client_id
            self.selected_client_id = (self.selected_client_id +
                                       1) % (len(self.clients_pool) + 1)
            if self.selected_client_id == 0:
                self.selected_client_id = 1
        selected_clients_list = []
        selected_clients_list.append(self.selected_client_id)
        self.selected_clients = None
        self.selected_clients = deepcopy(selected_clients_list)
        # starting time of a gloabl training round
        self.round_start_time = time.time()
        return self.selected_clients

    def load_gradients(self):
        """ Loading gradients from a file. """
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        model_path = f'{model_dir}{model_name}_gradients.pth'
        logging.info("[Server #%d] Loading gradients from %s.", os.getpid(),
                     model_path)

        return torch.load(model_path)

    async def client_payload_done(self, sid, client_id):
        assert self.client_payload[sid] is not None
        payload_size = 0
        if isinstance(self.client_payload[sid], list):
            for _data in self.client_payload[sid]:
                payload_size += sys.getsizeof(pickle.dumps(_data))
        else:
            payload_size = sys.getsizeof(pickle.dumps(
                self.client_payload[sid]))

        logging.info(
            "[Server #%d] Received %s MB of payload data from client #%d.",
            os.getpid(), round(payload_size / 1024**2, 2), client_id)
        
        # if clients send features, train it and return gradient
        if self.reports[sid].phase == "features":
            logging.info(
                "[Server #%d] client #%d features received. Processing.",
                os.getpid(), client_id)
            features = [self.client_payload[sid]]
            feature_dataset = list(chain.from_iterable(features))
            sampler = all_inclusive.Sampler(feature_dataset)
            self.algorithm.train(feature_dataset, sampler,
                             Config().algorithm.cut_layer)
            # Test the updated model
            self.accuracy = self.trainer.test(self.testset)
            logging.info('[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(os.getpid(), 100 * self.accuracy))
            
            payload = self.load_gradients()
            logging.info("[Server #%d] Reporting gradients to client #%d.",
                         os.getpid(), client_id)
            
            sid = self.clients[client_id]['sid']
            # payload = await self.customize_server_payload(pickle.dumps(payload))
            # Sending the server payload to the clients
            payload = self.load_gradients()
            await self.send(sid, payload, client_id)
            return

        self.updates.append((self.reports[sid], self.client_payload[sid]))
        
        if len(self.updates) > 0 and len(self.updates) >= len(
                self.selected_clients):
            logging.info(
                "[Server #%d] All %d client reports received. Processing.",
                os.getpid(), len(self.updates))
            await self.process_reports()
            await self.wrap_up()
            await self.select_clients()

    
