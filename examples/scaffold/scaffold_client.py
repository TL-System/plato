"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

import logging
import os

import pickle

from plato.clients import simple
from plato.config import Config
import extract_server_payload, send_extra_payload

class Client(simple.Client):
    """A SCAFFOLD federated learning client who sends weight updates
    and client control variate."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None, inbound_processor=None, outbound_processor=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.client_control_variate = None
        self.customized_processors = {'extract_server_payload': extract_server_payload.Processor, 'send_extra_payload':send_extra_payload.Processor}


    def process_server_response(self, server_response):
        """Initialize the server control variate and client control variate for trainer"""

        # Load the client control variate if the client has participated before
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_control_variate.pth"
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
