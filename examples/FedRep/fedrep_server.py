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

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)

        self.representation_param_names = []

    def extract_representation_param_names(self):
        """ Obtain the weights responsible for representation. """

        model_full_parameter_names = list(
            self.trainer.model.state_dict().keys())
        # in general, the weights before the final layer are regarded as
        #   the representation.
        # then the final layer is the last two key in the obtained key list.
        # For example,
        #  lenet:
        #  ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'fc4.weight', 'fc4.bias', 'fc5.weight', 'fc5.bias']
        #  representaion:
        #   ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 'fc4.weight', 'fc4.bias']
        self.representation_param_names = model_full_parameter_names[:-2]

        logging.info("Representation_weights: %s",
                     self.representation_param_names)

    def load_trainer(self):
        """ rewrite the load_trainer func to further extract the representaion keys """
        super().load_trainer()

        self.extract_representation_param_names()

        # the representation keys are regarded as the global model
        #   this needs to be set in the trainer for training the
        #   global and local model in the FedRep's way
        self.trainer.set_representation_and_head(
            representation_param_names=self.representation_param_names)

        self.algorithm.set_representation_param_names(
            representation_param_names=self.representation_param_names)

    async def customize_server_response(self, server_response):
        """ 
            The FedRep server sends parameter names belonging to the representation
            layers back to the clients.
        """
        # server sends the required the representaion to the client
        server_response[
            "representation_param_names"] = self.representation_param_names
        return server_response
