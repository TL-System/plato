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

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.representation_param_names = []

    def process_server_response(self, server_response) -> None:
        """Additional client-specific processing on the server response."""
        super().process_server_response(server_response)

        self.representation_param_names = server_response[
            "representation_param_names"]

        # The representation keys are regarded as the global model.
        # This needs to be set in the trainer for training the
        # global and local model according to the FedRep algorithm.
        self.trainer.set_representation_and_head(
            representation_param_names=self.representation_param_names)

        self.algorithm.set_representation_param_names(
            representation_param_names=self.representation_param_names)
