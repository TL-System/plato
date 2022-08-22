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

import pickle

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A SCAFFOLD federated learning client who sends weight updates
    and client control variate."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

        self.client_control_variate = None

        # Save the global model weights for computing new control variate
        # using the Option 2 in the paper
        self.global_model_weights = OrderedDict()

    def process_server_response(self, server_response):
        """Initialize the server control variate and client control variate for trainer"""

        # Load the client control variate if the client has participated before
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}_control_variate.pth"
        client_control_variate_path = f"{model_path}/{filename}"

        if os.path.exists(client_control_variate_path):
            logging.info(
                "[Client #%d] Loading the control variate from %s.",
                self.client_id,
                client_control_variate_path,
            )
            with open(client_control_variate_path, "rb") as path:
                self.client_control_variate = pickle.load(path)
            self.trainer.client_control_variate = self.client_control_variate
            logging.info("[Client #%d] Loaded the control variate.", self.client_id)
            self.trainer.extra_payload_path = client_control_variate_path

        # Create a copy of the global model weights prior to training
        for name, weight in self.trainer.model.cpu().state_dict().items():
            self.global_model_weights[name] = weight
        self.trainer.global_model_weights = self.global_model_weights

    def load_payload(self, server_payload):
        """Load model weights and server control variate from server payload onto this client."""
        self.algorithm.load_weights(server_payload[0])
        self.trainer.server_control_variate = server_payload[1]
