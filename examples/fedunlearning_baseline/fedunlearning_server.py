"""
A customized server for federated unlearning.

Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid
Retraining," in Proc. INFOCOM, 2022.

Reference: https://arxiv.org/abs/2203.07320
"""
import logging
import os
import pickle
import random
import time

import numpy as np
from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """ A federated unlearning server that implements the federated unlearning baseline algorithm.
    
    When we reach the 'data_deletion_round,' the server will roll back to the round, which is the minimum of the client_requesting_deletion first selected for the training.
    
    For example, if client[1] wants to delete its data after round 2, the server first finishes the aggregation at round 2, then finds out if or not the client[1] was selected in the previous round. If it was, roll back to the round that is the first time that client[1] is selected and start retraining. Otherwise, keep training with the client[1] and delete its data by all data * 'deleted_data_ratio.'
    """

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.restarted_session = True
        # Store client_id as keys, the round that corresponding client first be selected as values
        self.clients_arrive_round = {}
        self.retrain_phase = False
        # If current round is in retrain phase, retrain_phase becomes True
        self.start_retrain_round = []

    async def select_clients(self, for_next_batch=False):
        await super().select_clients(for_next_batch)

        for client_id in self.selected_clients:
            if not client_id in self.clients_arrive_round:
                self.clients_arrive_round[client_id] = self.current_round

    async def register_client(self, sid, client_id):
        """ Adding a newly arrived client to the list of clients. """
        if not client_id in self.clients:
            # The last contact time is stored for each client
            self.clients[client_id] = {
                'sid': sid,
                'last_contacted': time.perf_counter()
            }
            logging.info("[%s] New client with id #%d arrived.", self,
                         client_id)
        else:
            self.clients[client_id]['last_contacted'] = time.perf_counter()
            logging.info("[%s] New contact from Client #%d received.", self,
                         client_id)

        if (self.current_round == 0 or self.resumed_session) and len(
                self.clients) >= self.clients_per_round:
            logging.info("[%s] Starting training.", self)
            self.resumed_session = False
            # Saving a checkpoint for round #0 before any training starts,
            # useful if we need to roll back to the very beginning, such as
            # in the federated unlearning process
            self.save_to_checkpoint()
            await self.select_clients()

    async def wrap_up_processing_reports(self):
        """ Wrap up processing the reports with any additional work. """
        await super().wrap_up_processing_reports()

        are_clients_selected_before_retrain = any([
            client_id in self.clients_arrive_round
            for client_id in Config().clients.client_requesting_deletion
        ])

        if not are_clients_selected_before_retrain:
            logging.info(
                "[%s] Clients are not selected before data_deletion_round.",
                Config().clients.client_requesting_deletion)

        elif (self.current_round == Config().clients.data_deletion_round
              ) and self.restarted_session:
            # If data_deletion_round equals to the current round at server at the first time
            # retrain phase start
            self.retrain_phase = True

            for client_to_delete in Config(
            ).clients.client_requesting_deletion:
                if client_to_delete in self.clients_arrive_round:
                    self.start_retrain_round.append(
                        self.clients_arrive_round[client_to_delete])

            self.current_round = min(self.start_retrain_round) - 1
            self.restarted_session = False

            logging.info(
                "[%s] Data deleted. Retraining from the states after round #%s.",
                self, self.current_round)

            # Loading the saved model on the server for resuming the training session from round 1
            checkpoint_dir = Config.params['checkpoint_dir']

            model_name = Config().trainer.model_name if hasattr(
                Config().trainer, 'model_name') else 'custom'
            filename = f"checkpoint_{model_name}_{self.current_round}.pth"
            self.trainer.load_model(filename, checkpoint_dir)
            logging.info(
                "[Server #%d] Model used for the retraining phase loaded from %s.",
                os.getpid(), checkpoint_dir)

            if hasattr(Config().clients,
                       'exact_retrain') and Config().clients.exact_retrain:
                # Loading the PRNG states on the server in preparation for the retraining phase
                logging.info(
                    "[Server #%d] Random states after round #%s restored for exact retraining.",
                    os.getpid(), self.current_round)
                self.restore_random_states(self.current_round, checkpoint_dir)

    async def customize_server_response(self, server_response):
        """ Wrap up generating the server response with any additional information. """
        server_response['current_round'] = self.current_round
        server_response['retrain_phase'] = self.retrain_phase
        return server_response
