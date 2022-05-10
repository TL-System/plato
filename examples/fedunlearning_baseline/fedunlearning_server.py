"""
A customized server for federated unlearning.

Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid
Retraining," in Proc. INFOCOM, 2022.

Reference: https://arxiv.org/abs/2203.07320
"""
import logging
import time

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """ A federated unlearning server that implements the federated unlearning baseline 
        algorithm.
    """

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.restarted_session = True

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

        if (self.current_round == Config().clients.data_deletion_round
            ) and self.restarted_session:
            logging.info("[%s] Data deleted. Retraining from the first round.",
                         self)
            self.current_round = 0
            self.restarted_session = False

            # Loading the saved model the server for resuming the training session from round 1
            checkpoint_dir = Config.params['checkpoint_dir']

            model_name = Config().trainer.model_name if hasattr(
                Config().trainer, 'model_name') else 'custom'
            filename = f"checkpoint_{model_name}_{self.current_round}.pth"
            self.trainer.load_model(filename, checkpoint_dir)

    async def customize_server_response(self, server_response):
        """ Wrap up generating the server response with any additional information. """
        server_response['current_round'] = self.current_round
        return server_response
