"""
An implementation of the FedPer algorithm.

Manoj Ghuhan Arivazhagan, et.al, Federated learning with personalization layers, 2019.
https://arxiv.org/abs/1912.00818

Official code: None
Third-part code: https://github.com/jhoon-oh/FedBABU

We should note the FedPer is similar to the FedBABU as they both 
exchange the body of one model between the server and clients.

The difference is that during the local update, the FedPer will optimize 
the whole model while FedBABU only optimizes the body part.

"""


import logging

from examples.pfl.bases import simple_personalized
from plato.config import Config
from plato.utils import fonts


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

        # in FedPer, the head of personalized model should be assigned to the
        # received model to perform normal training.
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

        # therefore, everytime the client performs local update, the head of its initial
        # personalized model is assigned to the self.model, making:
        # the final global parameter and the initialized global parameter have the
        # same head parameter. See page 6 of the paper.
        # load the model
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
