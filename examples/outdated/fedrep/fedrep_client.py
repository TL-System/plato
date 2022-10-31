"""
A personalized federated learning client using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""

from plato.clients import simple


class Client(simple.Client):
    """A personalized federated learning client using the FedRep algorithm."""

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

    def process_server_response(self, server_response) -> None:
        """Additional client-specific processing on the server response."""
        super().process_server_response(server_response)

        # obtain the representation sent by the server
        self.representation_param_names = server_response["representation_param_names"]

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
