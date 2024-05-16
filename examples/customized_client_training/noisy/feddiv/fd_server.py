"""
FedDiv Server

"""

import random
import torch
import logging
import numpy as np

from plato.servers import fedavg
from plato.config import Config
from feddiv.gmm_filter import GlobalFilterManager
from plato.utils import fonts


class Server(fedavg.Server):
    """A MaskCrypt server with selective homomorphic encryption support."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)

        # Warm up
        self.warm_up = True
        self.warm_up_clients = []
        self.current_warm_up_round = 0
        self.warm_up_rounds = Config().server.feddiv.warm_up_rounds
        
        self.clients_per_round_bak = Config().clients.per_round
        # Normal training
        self.global_filter = None

    def configure(self):
        """Overide the configure function to setup noise filter"""
        super().configure()
        self.global_filter = GlobalFilterManager(
            components=2, seed=None, init_params="random"
        )

    def choose_clients(self, clients_pool, clients_count):
        """Choose clients with no replacement in warm up phase."""
        # Resume the per round client number
        self.clients_per_round = self.clients_per_round_bak
        clients_count = self.clients_per_round


        if not len(self.warm_up_clients) and self.warm_up:
            self.current_warm_up_round += 1
            if self.current_warm_up_round > self.warm_up_rounds:
                self.warm_up = False
                logging.info(fonts.colourize(f"[{self}] FedDiv warm up phase ends."))
            else:
                self.warm_up_clients = clients_pool[:]
                random.shuffle(self.warm_up_clients)

        if self.warm_up:
            selected_clients = self.warm_up_clients[:clients_count]
            self.warm_up_clients = self.warm_up_clients[clients_count:]
            self.clients_per_round = len(selected_clients)
            return selected_clients
        else:
            return super().choose_clients(clients_pool, clients_count)

    def customize_server_payload(self, payload):
        """Customize the server payload before sending to the client."""
        payload = {
            "warm_up": self.warm_up,
            "payload": payload,
        }
        if not self.warm_up:
            payload["filter"] = {
                "n_components": self.global_filter.components,
                "random_state": self.global_filter.random_state,
                "is_quiet": True,
                "init_params": self.global_filter.init_params,
                "weights_init": self.global_filter.model.weights_,
                "means_init": self.global_filter.model.means_,
                "precisions_init": self.global_filter.model.precisions_,
                "covariances_init": self.global_filter.model.covariances_,
            }

        return payload

    def weights_received(self, weights_received):
        if self.warm_up:
            # Directly return the model weights in warm up phase
            return weights_received
        else:
            data_sizes = [x["data_size"] for x in weights_received]
            filter_updates = [x["filter_updates"] for x in weights_received]
            self.global_filter.set_parameters_from_clients_models(filter_updates)
            self.global_filter.weighted_average_clients_models(data_sizes)
            self.global_filter.update_server_model()
            model_weights = [x["model_weights"] for x in weights_received]
            return model_weights
