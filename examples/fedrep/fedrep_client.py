"""
Implement the client for Fedrep method.

"""

import collections

from plato.config import Config
from plato.clients import simple
from plato.clients import base


class Client(simple.Client):
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.model_representation_weights_key = []

    def process_server_response(self, server_response) -> None:
        """Additional client-specific processing on the server response."""
        self.model_representation_weights_key = server_response[
            "representation_keys"]

        # the representation keys are regarded as the global model
        #   this needs to be set in the trainer for training the
        #   global and local model in the FedRep's way
        self.trainer.set_global_local_weights_key(
            global_keys=self.model_representation_weights_key)

    async def train(self):
        """ Implement the train for FedRep method. """
        report, weights = await super().train()

        # extract the representation weights as the global model
        representation_weights = collections.OrderedDict()
        for name, para in weights.items():
            if name in self.model_representation_weights_key:
                representation_weights[name] = para

        print("representation_weights: ", representation_weights.keys())

        return report, representation_weights