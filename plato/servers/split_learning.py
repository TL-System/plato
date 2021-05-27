"""
A split learning server.
"""

import logging
import os
import pickle
import time
from copy import deepcopy
from itertools import chain

import torch
from plato.config import Config
from plato.samplers import all_inclusive
from plato.servers import fedavg


class Server(fedavg.Server):
    """The split learning server."""
    def __init__(self):
        super().__init__()
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
                                       1) % (len(self.clients) + 1)
            if self.selected_client_id == 0:
                self.selected_client_id = 1
        selected_clients_list = []
        selected_clients_list.append(str(self.selected_client_id))
        self.selected_clients = None
        self.selected_clients = deepcopy(selected_clients_list)
        # starting time of a gloabl training round
        self.round_start_time = time.time()

    async def process_reports(self):
        """Process the features extracted by the client and perform server-side training."""
        features = [features for (__, features) in self.updates]

        # Faster way to deep flatten a list of lists compared to list comprehension
        feature_dataset = list(chain.from_iterable(features))

        # Training the model using all the features received from the client
        sampler = all_inclusive.Sampler(feature_dataset)
        self.algorithm.train(feature_dataset, sampler,
                             Config().algorithm.cut_layer)

        # Test the updated model
        self.accuracy = self.trainer.test(self.testset)
        logging.info('[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
            os.getpid(), 100 * self.accuracy))

        await self.wrap_up_processing_reports()

    async def wrap_up(self):
        """ Wrapping up when each round of training is done. """

        # Report gradients to client
        payload = self.load_gradients()
        if len(payload) > 0:
            client_id = str(self.selected_client_id)
            logging.info("[Server #%d] Reporting gradients to client #%d.",
                         os.getpid(), client_id)
            server_response = {
                'id': client_id,
                'payload': True,
                'payload_length': len(payload)
            }
            server_response = await self.customize_server_response(
                server_response)
            # Sending the server response as metadata to the clients (payload to follow)
            socket = self.clients[client_id]
            await socket.send(pickle.dumps(server_response))

            payload = await self.customize_server_payload(payload)

            # Sending the server payload to the clients
            await self.send(socket, payload)

            # Wait until client finish its train
            report = await self.clients[str(self.selected_client_id)].recv()
            payload = await self.clients[str(self.selected_client_id)].recv()

        # Break the loop when the target accuracy is achieved
        target_accuracy = Config().trainer.target_accuracy

        if target_accuracy and self.accuracy >= target_accuracy:
            logging.info("[Server #%d] Target accuracy reached.", os.getpid())
            await self.close()

        if self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")
            await self.close()

    def load_gradients(self):
        """ Loading gradients from a file. """
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        model_path = f'{model_dir}{model_name}_gradients.pth'
        logging.info("[Server #%d] Loading gradients from %s.", os.getpid(),
                     model_path)

        return torch.load(model_path)
