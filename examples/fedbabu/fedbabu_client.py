"""
An implementation of the FedBABU algorithm.

J. Oh, et al., "FedBABU: Toward Enhanced Representation for Federated Image Classification,"
in the Proceedings of ICLR 2022.

https://openreview.net/pdf?id=HuaYQfggn5u

Source code: https://github.com/jhoon-oh/FedBABU
"""


import logging

from plato.clients import simple_personalized
from plato.config import Config


class Client(simple_personalized.Client):
    """A FedBABU federated learning client."""

    def _load_payload(self, server_payload) -> None:
        """Load the server model onto this client.

        Each client will
        1. recevie the global model (body)
        2. load the personalized locally
        The received body and the extracted head of personalized mdoel
        will be combined to be assigned to the self.model for federated
        training.
        """
        logging.info(
            "[Client #%d] Received the global model (body) containing modules: %s.",
            self.client_id,
            self.algorithm.extract_modules_name(list(server_payload.keys())),
        )

        # in FedBABU, the head of one model is not trained during the federated
        # training stage, thus every time the client is selected, the initial
        # personalized model will be loaded to be assigned to the self.model
        # for federated training.
        self.persist_initial_personalized_model()
        # load the personalized model.
        self.load_personalized_model()

        # get the `head` from the personalized model head
        head_modules_name = Config().trainer.head_modules_name
        model_head_params = self.algorithm.extract_weights(
            model=self.personalized_model, modules_name=head_modules_name
        )
        logging.info(
            "[Client #%d] Extracted head modules: %s from its loaded personalized model.",
            self.client_id,
            self.algorithm.extract_modules_name(list(model_head_params.keys())),
        )
        server_payload.update(model_head_params)
        logging.info(
            "[Client #%d] Combined head modules to received modules.", self.client_id
        )
        # load the model
        self.algorithm.load_weights(server_payload)
