"""
A split learning server.
"""

import logging
import os

import torch
from plato.config import Config
from plato.datasources import feature
from plato.samplers import all_inclusive
from plato.servers import fedavg


class Server(fedavg.Server):
    """The split learning server."""

    def load_gradients(self):
        """Loading gradients from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        model_gradients_path = f"{model_path}/{model_name}_gradients.pth"
        logging.info("[Server #%d] Loading gradients from %s.", os.getpid(),
                     model_gradients_path)

        return torch.load(model_gradients_path)

    def customize_server_payload(self, payload):
        """ Wrap up generating the server payload with any additional information. """
        # sending global model to the clients
        return (payload, 'weights')

    async def process_client_info(self, client_id, sid):
        """Process the received metadata information from a reporting client."""
        if self.reports[sid].phase == "features":
            payload = self.client_payload[sid]
            feature_dataset = feature.DataSource(payload)

            # Training the model using all the features received from the client
            sampler = all_inclusive.Sampler(feature_dataset)
            self.algorithm.train(feature_dataset, sampler)

            # Sending the gradients calculated by the server to the clients
            gradients = self.load_gradients()
            logging.info("[Server #%d] Reporting gradients to client #%d.",
                         os.getpid(), client_id)
            server_payload = (gradients, 'gradients')
            await self.send(sid, server_payload, client_id)

        elif self.reports[sid].phase == "weights":
            await super().process_client_info(client_id, sid)