"""
Implement the server for Fedrep method.

"""

import logging
from plato.servers import fedavg


class Server(fedavg.Server):

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)

        self.model_representation_weights_key = []

    def extract_representation_weights_key(self):
        """ Obtain the weights responsible for representation. """

        model_full_weights_key = list(self.trainer.model.state_dict().keys())
        # in general, the weights before the final layer are regarded as
        #   the representation.
        # then the final layer is the last two key in the obtained key list.
        # For example,
        #  lenet:
        #  ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'fc4.weight', 'fc4.bias', 'fc5.weight', 'fc5.bias']
        #  representaion:
        #   ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'fc4.weight', 'fc4.bias']
        self.model_representation_weights_key = model_full_weights_key[:-2]

        logging.info("Representation_weights: %s",
                     self.model_representation_weights_key)

    def load_trainer(self):
        """ rewrite the load_trainer func to further extract the representaion keys """
        super().load_trainer()

        self.extract_representation_weights_key()

        # the representation keys are regarded as the global model
        #   this needs to be set in the trainer for training the
        #   global and local model in the FedRep's way
        self.trainer.set_global_local_weights_key(
            global_keys=self.model_representation_weights_key)

        self.algorithm.set_global_weights_key(
            global_keys=self.model_representation_weights_key)

    async def customize_server_response(self, server_response):
        """ Wrap up generating the server response with any additional information. """
        # server sends the required the representaion to the client
        server_response[
            "representation_keys"] = self.model_representation_weights_key
        return server_response
