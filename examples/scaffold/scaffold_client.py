"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

from collections import OrderedDict
import logging
import os

import torch

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A SCAFFOLD federated learning client who sends weight updates
    and client control variate."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

        self.client_control_variate = None
        self.server_control_variate = None
        self.control_variate_path = None

        # Save the global model weights for computing new control variate
        # using the Option 2 in the paper
        self.global_model_weights = OrderedDict()

    async def train(self):
        """ Initialize the server control variate and client control variate for trainer. """
        # Load the client's control variate if it has participated before
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}_control_variate.pth"
        self.control_variate_path = f"{model_dir}/{filename}"

        if os.path.exists(self.control_variate_path):
            logging.info("[Client #%d] Loading the control variate from %s.",
                         self.client_id, self.control_variate_path)
            self.client_control_variate = torch.load(self.control_variate_path)
            self.trainer.client_control_variate = self.client_control_variate
            logging.info("[Client #%d] Loaded the control variate.",
                         self.client_id)

        self.trainer.server_control_variate = self.server_control_variate

        for name, weight in self.trainer.model.cpu().state_dict().items():
            self.global_model_weights[name] = weight

        report, weights = await super().train()

        # Compute deltas of this client's control variate
        new_client_control_variate = OrderedDict()
        control_variate_deltas = OrderedDict()
        if self.client_control_variate:
            for name, previous_weight in self.global_model_weights.items():
                new_client_control_variate[name] = torch.sub(
                    self.client_control_variate[name],
                    self.server_control_variate[name])
                new_client_control_variate[name].add_(
                    torch.sub(previous_weight,
                              self.algorithm.extract_weights()[name]),
                    alpha=1 / Config().trainer.epochs)

                control_variate_deltas[name] = torch.sub(
                    new_client_control_variate[name],
                    self.client_control_variate[name])
        else:
            for name, previous_weight in self.global_model_weights.items():
                new_client_control_variate[
                    name] = -self.server_control_variate[name]
                new_client_control_variate[name].add_(
                    torch.sub(previous_weight,
                              self.algorithm.extract_weights()[name]),
                    alpha=1 / Config().trainer.epochs)

                control_variate_deltas[name] = new_client_control_variate[name]

        # Update client control variate
        self.client_control_variate = new_client_control_variate

        # Save client control variate
        logging.info("[Client #%d] Saving the control variate to %s.",
                     self.client_id, self.control_variate_path)
        torch.save(self.client_control_variate, self.control_variate_path)
        logging.info("[Client #%d] Control variate saved to %s.",
                     self.client_id, self.control_variate_path)

        return report, [weights, control_variate_deltas]

    def load_payload(self, server_payload):
        """Load model weights and server control variate from server payload onto this client."""
        self.algorithm.load_weights(server_payload[0])
        self.server_control_variate = server_payload[1]
