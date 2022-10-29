"""
A personalized federated learning server using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

import logging

from plato.servers import fedavg


class Server(fedavg.Server):
    """A personalized federated learning server using the FedRep algorithm."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # parameter names of the representation
        #   As mentioned by Eq. 1 and Fig. 2 of the paper, the representation
        #   behaves as the global model.
        self.representation_param_names = []

    def extract_representation_param_names(self):
        """Obtain the weights responsible for representation."""

        model_full_parameter_names = list(self.trainer.model.state_dict().keys())

        # in general, the weights before the final layer are regarded as
        #   the representation.
        # then the final layer is regarded as the head.
        # For example,
        #  lenet:
        #  ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
        #  'conv3.weight', 'conv3.bias', 'fc4.weight', 'fc4.bias',
        #  'fc5.weight', 'fc5.bias']
        #  representaion:
        #  ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
        #  'conv3.weight', 'conv3.bias', 'fc4.weight', 'fc4.bias']
        self.representation_param_names = model_full_parameter_names[:-2]

        logging.info("Representation_weights: %s", self.representation_param_names)

    def init_trainer(self) -> None:
        """Extract representation parameter names after initializing the trainer."""
        super().init_trainer()

        self.extract_representation_param_names()

        # The trainer responsible for optimizing the model should know
        # which part parameters behave as the representation and which
        # part of the parameters behave as the head. The main reason is
        # that the head is optimized in the 'Client Update' while the
        # representation is optimized in the 'Server Update', as mentioned
        # in Section 3 of the FedRep paper.
        self.trainer.set_representation_and_head(
            representation_param_names=self.representation_param_names
        )

        # The algorithm only operates on the representation without
        # considering the head as the head is solely known by each client
        # because of personalization.
        self.algorithm.set_representation_param_names(
            representation_param_names=self.representation_param_names
        )

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """
        The FedRep server sends parameter names belonging to the representation
        layers back to the clients.
        """
        # server sends the required representaion to the client
        server_response["representation_param_names"] = self.representation_param_names
        return server_response
