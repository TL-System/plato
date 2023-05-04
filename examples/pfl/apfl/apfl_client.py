"""
The implementation of APFL method based on the plato's pFL code.

Yuyang Deng, et.al, Adaptive Personalized Federated Learning

paper address: https://arxiv.org/abs/2001.01523

Official code: None
Third-part code: 
- https://github.com/lgcollins/FedRep
- https://github.com/MLOPTPSU/FedTorch/blob/main/main.py
- https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py

"""

import logging

from examples.pfl.bases import simple_personalized


class Client(simple_personalized.Client):
    """A APFL federated learning client."""

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
        # in APFL, the personalized model is trained together with the
        # global model
        # thus, personalized model should be loaded.
        self.persist_initial_personalized_model()
        # load the personalized model.
        loaded_status = self.load_personalized_model()

        self.trainer.extract_alpha(loaded_status)

        # assign the received payload to the local model
        self.algorithm.load_weights(server_payload)
