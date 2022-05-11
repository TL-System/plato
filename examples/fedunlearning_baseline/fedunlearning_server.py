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

def decode_config_with_comma(target_string):
    if isinstance(target_string, int):
        return [target_string]       
    else: 
        return list(map(lambda x: int(x), target_string.split(", ")))

class Server(fedavg.Server):
    """ A federated unlearning server that implements the federated unlearning baseline 
        algorithm.
    """

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)
        self.restarted_session = True
        self.clients_dic = {}

    async def select_clients(self, for_next_batch=False):
        await super().select_clients(for_next_batch)

        for client_id in self.selected_clients:
            if not client_id in self.clients_dic.keys():
                self.clients_dic[client_id] = self.current_round

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
        
        client_requesting_deletion = decode_config_with_comma(Config().clients.client_requesting_deletion)

        are_clients_selected_before_retrain = any([client_id in self.clients_dic.keys() for client_id in client_requesting_deletion])
        print(self.clients_dic)
        print("-------")
        print(client_requesting_deletion)
        print("-------")
        print([client_id in self.clients_dic.keys() for client_id in client_requesting_deletion])
        
        if not are_clients_selected_before_retrain: 
             logging.info("[%s] Clients are not selected before data_deletion_round.", client_requesting_deletion)

        elif (self.current_round == Config().clients.data_deletion_round
            ) and self.restarted_session:
            logging.info("[%s] Data deleted. Retraining from the first round.",
                         self)
            print("self.current_round: ", self.current_round)     
            client_requesting_deletion = decode_config_with_comma(Config().clients.client_requesting_deletion)
            start_retrain_round = [self.clients_dic[client_id] for client_id in client_requesting_deletion if client_id in self.clients_dic.keys()]
            initial_checkpoint_round = min(start_retrain_round)
            self.restarted_session = False
            print("*************************")
            print(client_requesting_deletion)
            print(start_retrain_round)
            print("initial_checkpoint_round: ", initial_checkpoint_round)
            # Loading the saved model the server for resuming the training session from round 1
            checkpoint_dir = Config.params['checkpoint_dir']

            model_name = Config().trainer.model_name if hasattr(
                Config().trainer, 'model_name') else 'custom'
            filename = f"checkpoint_{model_name}_{initial_checkpoint_round}.pth"
            self.trainer.load_model(filename, checkpoint_dir)
            # The function select_clients() in server/base.py will add 1 to current_round
            self.current_round = initial_checkpoint_round - 1
            print("self.current_round: ", self.current_round)
        else:
            pass

    async def customize_server_response(self, server_response):
        """ Wrap up generating the server response with any additional information. """
        server_response['current_round'] = self.current_round
        return server_response
