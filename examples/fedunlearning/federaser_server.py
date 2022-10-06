"""
A customized server for federaser

Liu et al., "FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning
Models," in 2021 IEEE/ACM 29th International Symposium on Quality of Service (IWQoS 2021).

Reference: https://ieeexplore.ieee.org/document/9521274
"""

import torch
import numpy as np
from plato.config import Config

import fedunlearning_server
#import mia


class Server(fedunlearning_server.Server):
    """ A federated unlearning server that implements the FedEraser Algorithm.

    Retains the clients update each round for later calibration use.
    Exclude the forgotten clients' contribution during retraining.
    Leverage the calibration training update to calibrate the retained updates.
    """

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # Retain a list of client updates each round for later calibration
        self.retained_client_updates = []

        # The attack model for MIA
        self.attack_model = None

        self.updated_epochs = None
        # DataLoaders for MIA
        # (might be inappropriate to put in the server since the server shouldn't get the data)
        self.client_loaders = None
        self.test_loader = None

        self.next_round = 0   

    async def aggregate_weights(self, updates):
        """
        Retain client updates when not retraining.
        Leverage the current update to calibrate retained updates when retraining
        """
        if not self.retraining:
            # Retain client updates for calibration (list index = current round - 1)
            self.retained_client_updates.append(self.updates)

        if (self.current_round == Config().clients.data_deletion_round
            ) and not self.retraining and not hasattr(Config().server,"clusters"):
            updated_epochs = np.ceil(
                    Config().trainer.epochs *
                    Config().clients.forget_local_epoch_ratio)
            self.updated_epochs = int(updated_epochs)

        elif self.retraining and self.current_round < (Config().clients.data_deletion_round - ((Config().clients.data_deletion_round - 1) % Config().clients.delta_t)):
            clients_to_delete = Config().clients.clients_requesting_deletion
            for (client_id_cali, __, payload_cali, __) in updates:
                if client_id_cali in clients_to_delete:
                    baseline_weights = self.algorithm.extract_weights()
                    # If the client is to be forgotten, let its weights equal to the baseline
                    # weights so it has zero contribution
                    for name, __ in payload_cali.items():
                        baseline = baseline_weights[name]
                        payload_cali[name] = baseline

                    continue

                for (client_id_retained, __, payload_retained,
                     __) in self.retained_client_updates[self.current_round -
                                                         1]:
                    if client_id_retained == client_id_cali:
                        # Leverage the current update to calibrate the retained updates
                        # new_update = |retained_client_update| *
                        # (calibration_update/||calibration_update||)
                        for name, current_weight in payload_cali.items():
                            payload_cali[name] = torch.norm(
                                payload_retained[name].float()
                            ) * (current_weight / torch.norm(current_weight.float()))
            updated_epochs = np.ceil(
                    Config().trainer.epochs *
                    Config().clients.forget_local_epoch_ratio)
            self.updated_epochs = int(updated_epochs)
            if self.current_round > Config().clients.data_deletion_round:
                self.updated_epochs = None
        
        else:
            self.updated_epochs = None

        await super().aggregate_weights(updates)

    async def wrap_up_processing_reports(self):
        # If the current round is the data deletion round and not yet in retrain phase,
        # train the MIA attack model using current model and attack the model before unlearning
        # if (self.current_round == Config().clients.data_deletion_round
        #     ) and not self.retraining:
        #     self.attack_model, self.client_loaders, self.test_loader = mia.train_attack_model(
        #         self.trainer.model, self.round_first_selected)

            #mia.attack(self.trainer.model, self.attack_model,self.client_loaders, self.test_loader)
        
        await super().wrap_up_processing_reports()
        
        if self.retraining and Config(
        ).server.federaser and Config().clients.delta_t != 1 and self.current_round < Config().clients.data_deletion_round:
            # Consider the retrain interval delta_t
         

            if self.next_round <= Config().clients.data_deletion_round:
            #     self.current_round = Config().clients.data_deletion_round + 1
            # else:
                self.current_round = self.next_round
            self.next_round = self.current_round + Config().clients.delta_t

            checkpoint_path = Config.params['checkpoint_path']
            self.restore_random_states(self.current_round, checkpoint_path)


        # Stop retraining if it reaches data_deletion_round again during retraining
        # since there's no need to do the calibration training afterwards
        # And do the MIA on the unlearned model
        # if (self.current_round
        #         == Config().clients.data_deletion_round) and self.retraining:
        #     self.retraining = False
        #     mia.attack(self.trainer.model, self.attack_model,
        #                self.client_loaders, self.test_loader)

    async def customize_server_response(self, server_response, client_id=None):
        """ Wrap up generating the server response with any additional information. """
        server_response['updated_epochs'] = self.updated_epochs 
        return server_response

