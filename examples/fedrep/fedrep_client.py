"""
Implement the client for Fedrep method.

"""

from plato.clients import simple


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

        self.algorithm.set_global_weights_key(
            global_keys=self.model_representation_weights_key)

    def load_payload(self, server_payload) -> None:
        """Loading the server model onto this client."""
        print("Loading server payload: ", server_payload.keys())
        self.algorithm.load_weights(server_payload)