"""
The client implementation of Ditto method based on the pFL framework of Plato.

Tian Li, et.al, Ditto: Fair and robust federated learning through personalization, 2021:
 https://proceedings.mlr.press/v139/li21h.html

Official code: https://github.com/s-huu/Ditto
Third-part code: https://github.com/lgcollins/FedRep

"""

import logging

from ..bases import simple_personalized


class Client(simple_personalized.Client):
    """A Ditto federated learning client."""

    def _load_payload(self, server_payload) -> None:
        """Load the server model onto this client.

        Each client will
        1. recevie the global model
        2. load the personalized locally

        """
        logging.info(
            "[Client #%d] Received the payload containing modules: %s.",
            self.client_id,
            self.algorithm.extract_modules_name(list(server_payload.keys())),
        )
        # in Ditto, the personalized model is trained together with the
        # global model
        # thus, personalized model should be loaded.
        self.persist_initial_personalized_model()
        # load the personalized model.
        self.load_personalized_model()

        # assign the received payload to the local model
        self.algorithm.load_weights(server_payload)
