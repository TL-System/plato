"""
A customized server for federated unlearning.

Reference: Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining." in Proc. INFOCOM, 2022 https://arxiv.org/abs/2203.07320

"""
import logging
import pickle
import torch
import torch.nn.functional as F

from plato.config import Config
from plato.servers import fedavg
import logging


class Server(fedavg.Server):
    """ A federated unlearning server that implements the federated unlearning baseline algorithm. """
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__()
        self.restarted_session = False

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

        if (self.current_round == 0 or self.resumed_session or not self.restarted_session) and len(
                self.clients) >= self.clients_per_round:
            logging.info("[%s] Starting training.", self)
            self.resumed_session = False
            self.restarted_session = False

            # Saving a checkpoint for round #0 before any training starts,
            # useful if we need to roll back to the very beginning, such as
            # in the federated unlearning process
            self.save_to_checkpoint()
            await self.select_clients()

    async def wrap_up_processing_reports(self):
        """ Wrap up processing the reports with any additional work. """
        await super().wrap_up_processing_reports()

        if self.current_round == Config().clients.data_deleted_round:
            logging.info("[%s] Data deleted. Retraining from the first round.", self)
            self.current_round = 0
            self.restarted_session = True

            # Loading the saved model the server for resuming the training session from round 1
            checkpoint_dir = Config.params['checkpoint_dir']

            model_name = Config().trainer.model_name if hasattr(
                Config().trainer, 'model_name') else 'custom'
            filename = f"checkpoint_{model_name}_{self.current_round}.pth"
            self.trainer.load_model(filename, checkpoint_dir)            