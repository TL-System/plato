"""
A split learning server.
"""

import logging
import os
import pickle
import sys
import time
from itertools import chain

import torch
from plato.config import Config
from plato.datasources import feature
from plato.samplers import all_inclusive
from plato.servers import fedavg


class Server(fedavg.Server):
    """The split learning server."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # a FIFO queue(list) for choosing the running client
        self.clients_running_queue = []

    def choose_clients(self, clients_pool, clients_count):
        assert len(clients_pool) > 0 and clients_count == 1

        # fist step: make sure that the sl running queue sync with the clients pool
        new_client_id_set = set(clients_pool)
        old_client_id_set = set(self.clients_running_queue)
        # delete the disconnected clients
        remove_clients = old_client_id_set - new_client_id_set
        for i in remove_clients:
            self.clients_running_queue.remove(i)
        # add the new registered clients
        add_clients = new_client_id_set - old_client_id_set
        for i in add_clients:
            insert_index = len(self.clients_running_queue) - 1
            self.clients_running_queue.insert(insert_index, i)

        # second step: use FIFO strategy to choose one client
        res_list = []
        if len(self.clients_running_queue) > 0:
            queue_head = self.clients_running_queue.pop(0)
            res_list.append(queue_head)
            self.clients_running_queue.append(queue_head)
            self.round_start_time = time.time()

        return res_list

    def load_gradients(self):
        """ Loading gradients from a file. """
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        model_path = f'{model_dir}{model_name}_gradients.pth'
        logging.info("[Server #%d] Loading gradients from %s.", os.getpid(),
                     model_path)

        return torch.load(model_path)

    async def client_payload_done(self, sid, client_id, s3_key=None):
        if s3_key is None:
            assert self.client_payload[sid] is not None

            payload_size = 0
            if isinstance(self.client_payload[sid], list):
                for _data in self.client_payload[sid]:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            else:
                payload_size = sys.getsizeof(
                    pickle.dumps(self.client_payload[sid]))
        else:
            self.client_payload[sid] = self.s3_client.receive_from_s3(s3_key)
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
            feature_dataset = feature.DataSource(features)
            sampler = all_inclusive.Sampler(feature_dataset)
            self.algorithm.train(feature_dataset, sampler,
                                 Config().algorithm.cut_layer)
            # Test the updated model
            self.accuracy = self.trainer.test(self.testset)
            logging.info(
                '[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
                    os.getpid(), 100 * self.accuracy))

            payload = self.load_gradients()
            logging.info("[Server #%d] Reporting gradients to client #%d.",
                         os.getpid(), client_id)

            sid = self.clients[client_id]['sid']
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
