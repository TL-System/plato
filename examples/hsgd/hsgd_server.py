"""
A cross-silo federated learning server that tunes
clients' local epoch numbers of each institution.
"""

import math
import torch
import logging
import asyncio
import random
import time
import multiprocess as mp
from plato.config import Config
from plato.servers import fedavg_cs
from plato.utils import csv_processor, fonts



class Server(fedavg_cs.Server):
    """
    A cross-silo federated learning server that tunes
    clients' local epoch numbers of each institution.
    """

    def __init__(self):
        super().__init__()

        # The central server uses a list to store each institution's clients' local epoch numbers
        self.local_epoch_list = None
        if Config().is_central_server():
            self.local_epoch_list = [
                Config().trainer.epochs for i in range(Config().algorithm.total_silos)
            ]
        self.result_file=None

    def clients_processed(self):
        """Additional work to be performed after client reports have been processed."""
        super().clients_processed()

        if Config().is_central_server():
            if self.result_file is None:
                self.result_file = f"{Config().params['result_path']}/central_server.csv"
                with open(self.result_file, 'w') as f:
                    f.write(','.join(self.get_logged_items().keys()) + "\n")
            with open(self.result_file, 'a') as f:
                f.write(','.join([str(v) for i, v in self.get_logged_items().items()]) + "\n")
        else:
            if self.result_file is None:
                self.result_file = f"{Config().params['result_path']}/edge_server_{self.trainer.client_id}.csv"
                with open(self.result_file, 'w') as f:
                    f.write(','.join(self.get_logged_items().keys()) + "\n")
            with open(self.result_file, 'a') as f:
                f.write(','.join([str(v) for i, v in self.get_logged_items().items()]) + "\n")
    
    def start_edge_servers(self, edge_server, edge_client, trainer):
        Server.start_clients(
            as_server=True,
            client=self.client,
            edge_server=edge_server,
            edge_client=edge_client,
            trainer=trainer,
        )

    def run(self, client=None, edge_server=None, edge_client=None, trainer=None):
        """Start a run loop for the server."""
        self.client = client
        self.configure()
        
        if Config().args.resume:
            self.resume_from_checkpoint()

        #Starting the global server process in a non-blocking way
        proc = mp.Process(
            target=Server.start_self,
            args=(self,)
            )
        proc.start()

        # In cross-silo FL, the central server lets edge servers start first
        # Then starts their clients
        # Tim: delegate client launch to "training_will_start()" to make sure they are 
        # started after the edge server

        self.start_edge_servers(edge_server, edge_client, trainer)
        
        # # Allowing some time for the edge servers to start
        # # Tim: add actual handshake here
        # time.sleep(10)

    def training_will_start(self):
        super().training_will_start()
        if Config().is_central_server():
            if self.disable_clients:
                logging.info("No clients are launched (server:disable_clients = true)")
            else:
                Server.start_clients(client=self.client)

    @staticmethod
    def start_self(self):

        asyncio.get_event_loop().create_task(self.periodic(self.periodic_interval))

        if hasattr(Config().server, "random_seed"):
            seed = Config().server.random_seed
            logging.info("Setting the random seed for selecting clients: %s", seed)
            random.seed(seed)
            self.prng_state = random.getstate()
        
        self.start()


