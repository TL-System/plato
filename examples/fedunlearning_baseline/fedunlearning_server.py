"""
A customized server for federated unlearning.

Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid
Retraining," in Proc. INFOCOM, 2022.

Reference: https://arxiv.org/abs/2203.07320
"""
import logging
import os
import time

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """ A federated unlearning server that implements the federated unlearning baseline algorithm.

    When 'data_deletion_round' specified in the configuration, the server will enter a retraining
    phase after this round is reached, during which it will roll back to the minimum round number
    necessary for all the clients requesting data deletion.

    For example, if client #1 wishes to delete its data after round #2, the server first finishes
    its aggregation at round #2, then finds out whether or not client #1 was selected in one of the
    previous rounds. If it was, the server will roll back to the round when client #1 was selected
    for the first time, and starts retraining phases from there. Otherwise, it will keep training
    but with client #1 deleting a percentage of its data samples, according to `delete_data_ratio`
    in the configuration.
    """

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        self.retraining = False

        # A dictionary that maps client IDs to the first round when the server selected it
        self.round_first_selected = {}

    async def select_clients(self, for_next_batch=False):
        """ Remembers the first round that a particular client ID was selected. """
        await super().select_clients(for_next_batch)

        for client_id in self.selected_clients:
            if not client_id in self.round_first_selected:
                self.round_first_selected[client_id] = self.current_round

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
        """ Enters the retraining phase if a specific set of conditions are satisfied. """
        await super().wrap_up_processing_reports()

        clients_to_delete = Config().clients.clients_requesting_deletion

        if (self.current_round == Config().clients.data_deletion_round
            ) and not self.retraining:
            # If data_deletion_round equals to the current round at server for the first time,
            # and the clients requesting retraining has been selected before, the retraining
            # phase starts
            earliest_round = self.current_round

            for client_id, first_round in self.round_first_selected.items():
                if client_id in clients_to_delete:
                    self.retraining = True

                    if earliest_round > first_round:
                        earliest_round = first_round

            if self.retraining:
                self.current_round = earliest_round - 1

                logging.info(
                    "[%s] Data deleted. Retraining from the states after round #%s.",
                    self, self.current_round)

                # Loading the saved model on the server for starting the retraining phase
                checkpoint_path = Config.params['checkpoint_path']

                model_name = Config().trainer.model_name if hasattr(
                    Config().trainer, 'model_name') else 'custom'
                filename = f"checkpoint_{model_name}_{self.current_round}.pth"
                self.trainer.load_model(filename, checkpoint_path)

                logging.info(
                    "[Server #%d] Model used for the retraining phase loaded from %s.",
                    os.getpid(), checkpoint_path)

                if hasattr(Config().clients,
                           'exact_retrain') and Config().clients.exact_retrain:
                    # Loading the PRNG states on the server in preparation for the retraining phase
                    logging.info(
                        "[Server #%d] Random states after round #%s restored for exact retraining.",
                        os.getpid(), self.current_round)
                    self.restore_random_states(self.current_round,
                                               checkpoint_path)
