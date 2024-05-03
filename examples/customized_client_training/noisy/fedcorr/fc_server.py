"""
FedCorr Server

"""

import random
import torch
import logging
import numpy as np

from plato.servers import fedavg
from plato.config import Config
from plato.utils import fonts
from sklearn.mixture import GaussianMixture


class Server(fedavg.Server):
    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)

        self.stage = 1
        self.random_seed = 1

        # Stage 1
        self.stage_1_rounds = Config().server.fedcorr.stage_1_rounds
        self.current_stage_1_round = 0
        self.non_selected_clients = []
        
        self.noisy_clients = []
        self.clean_clients = []
        self.estimated_noisy_level = []

        self.LID_accumulative_client = np.zeros(Config().clients.total_clients)

        # Stage 2
        self.clients_per_round_bak = Config().clients.per_round
        self.stage_2_rounds = Config().server.fedcorr.stage_2_rounds
        self.current_stage_2_round = 0

    def choose_clients(self, clients_pool, clients_count):
        """Choose clients with no replacement in warm up phase."""
        
        # Prepare the clients to be selected for current stage
        if self.stage == 1 and not len(self.non_selected_clients):
            self.current_stage_1_round += 1
            if self.current_stage_1_round > self.stage_1_rounds:
                self.stage = 2
                logging.info(fonts.colourize(f"[{self}] FedCorr Stage 1 ends."))
            else:
                self.non_selected_clients = clients_pool[:]
                random.shuffle(self.non_selected_clients)

        # Select client
        if self.stage == 1:
            selected_clients = self.non_selected_clients[:clients_count]
            self.non_selected_clients = self.non_selected_clients[clients_count:]
            return selected_clients
        elif self.stage == 2:
            self.clients_per_round = min(self.clients_per_round, len(self.clean_clients))
            self.current_stage_2_round += 1
            if self.current_stage_2_round <= self.stage_2_rounds:
                # Select clean clients for normal training
                logging.info(f"Clean clients:{self.clean_clients}")
                return super().choose_clients(list(self.clean_clients), min(clients_count, len(self.clean_clients)))
            else:
                logging.info(f"Noisy clients:{self.noisy_clients}")
                self.stage = 3
                self.clients_per_round = len(self.noisy_clients)
                return super().choose_clients(self.noisy_clients, len(self.noisy_clients))
        elif self.stage == 3:
            self.clients_per_round = self.clients_per_round_bak
            return super().choose_clients(clients_pool, self.clients_per_round)

    def customize_server_payload(self, payload):
        """Customize the server payload before sending to the client."""
        payload = {
            "stage": self.stage,
            "payload": payload,
        }
        if self.stage == 1 or self.stage == 2:
            payload["noisy_clients"] = self.noisy_clients
        else:
            payload["noisy_clients"] = None
        return payload

    def split_clients(self,):
        gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=self.random_seed).fit(
            np.array(self.LID_accumulative_client).reshape(-1, 1))
        labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(self.LID_accumulative_client).reshape(-1, 1))
        clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]

        self.noisy_clients = (np.where(labels_LID_accumulative != clean_label)[0] + 1).tolist()
        self.clean_clients = (np.where(labels_LID_accumulative == clean_label)[0] + 1).tolist()


    def weights_received(self, weights_received):
        if self.stage == 1:
            client_ids = np.array([x["client_id"] for x in weights_received]) - 1
            LID_clients = np.array([x["LID_client"] for x in weights_received])
            self.LID_accumulative_client[client_ids] = LID_clients

            if len(self.non_selected_clients) == 0:
                self.split_clients()

            model_weights = [x["model_weights"] for x in weights_received]
            return model_weights
        else:
            return weights_received
