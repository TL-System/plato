"""
An implementation of the LG-FedAvg algorithm.

Paul Pu Liang, et.al, Think Locally, Act Globally: Federated Learning with 
Local and Global Representations. 
https://arxiv.org/abs/2001.01523

Official code: https://github.com/pliang279/LG-FedAvg

"""


import logging

from examples.pfl.bases import simple_personalized
from plato.config import Config
from plato.utils import fonts


class Client(simple_personalized.Client):
    """A LG-FedAvg federated learning client."""

    def _load_payload(self, server_payload) -> None:
        """Load the server model onto this client.

        Each client will
        1. recevie the global model (head)
        2. load the personalized locally
        The received head and the extracted body of personalized mdoel
        will be combined to be assigned to the self.model for federated
        training.
        """
        logging.info(
            "[Client #%d] Received the global model (head) containing modules: %s.",
            self.client_id,
            self.algorithm.extract_modules_name(list(server_payload.keys())),
        )

        # in LG-FedAvg, the head of one model is not trained during the federated
        # training stage, thus every time the client is selected, the initial
        # personalized model will be loaded to be assigned to the self.model
        # for federated training.
        self.persist_initial_personalized_model()
        # load the personalized model.
        self.load_personalized_model()

        # get the `body` from the personalized model head
        body_modules_name = Config().trainer.body_modules_name
        model_body_params = self.algorithm.extract_weights(
            model=self.personalized_model, modules_name=body_modules_name
        )
        logging.info(
            "[Client #%d] Extracted body modules: %s from its loaded personalized model.",
            self.client_id,
            self.algorithm.extract_modules_name(list(model_body_params.keys())),
        )
        server_payload.update(model_body_params)
        logging.info(
            "[Client #%d] Combined head modules to received modules.", self.client_id
        )

        # assign the combination of the local and gloabl to self.model
        self.algorithm.load_weights(server_payload)

        if self.is_personalized_learn() and self.personalized_model is not None:
            # during the personalized learning, the received global modules will be
            # assigned to the self.personalized_model
            # the updated `server_payload` can be directly used here because this
            # the combination of the received global modules and the head of its
            # personalized model.
            self.personalized_model.load_state_dict(server_payload, strict=True)
            logging.info(
                fonts.colourize(
                    "[Client #%d] Assigned received global modules to its personalized model.",
                    colour="blue",
                ),
                self.client_id,
            )
