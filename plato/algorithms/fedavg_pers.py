"""
The federated averaging algorithm for self-supervised method.

The main target of this algorithm is to achieve the property of
randomly exchanging the whole defined or its sub-module between
the server and clients.
Thus, the 'extract_weights' functions extract the model' parameters
based on the required 'global_model_name'.

"""

import logging
from collections import OrderedDict

from plato.algorithms import fedavg
from plato.config import Config
from plato.trainers.base import Trainer


def is_global_parameter(param_name, global_model_name):
    """ whether the param_name in the desired global model name.

        e.g. the global_model_name is:
            - whole
            - online_network
            - online_network; online_predictor
    """
    flag = False
    global_params_prefix = global_model_name.split("__")
    for para_prefix in global_params_prefix:
        if para_prefix in param_name:
            flag = True

    return flag


class Algorithm(fedavg.Algorithm):
    """ Federated averaging algorithm for Byol models, used by both the client and the server. """

    def __init__(self, trainer: Trainer):
        super().__init__(trainer=trainer)
        # the sub-module's name that is used as the global model
        # shared among clients.
        # by default, the whole model will be used as the global model
        # i.e., global_model_name = "whole"
        # however, the user can set which component of the model is used
        # as the global model by setting its name. Thus, the algorithm can
        # extract the corresponding global sub-module from the whole model
        # to be exchanged between the server and clients
        self.global_model_name = "whole"

    def extract_weights(self, model=None):
        """Extract weights from the model.

            To not break the current Plato's code, we follow the principle:
            1.- Once the model is provided, i,e, model != None, its paramters
                will be extracted directly.
            2.- If the model is None, it is required to extract the default
                model, i.e., the self.model of this algo. Currenty, the
                'global_model_name' defined in the config file will be used
                to extract the desired sub-module from the self.model

        """
        if hasattr(Config().trainer, "global_model_name"):
            self.global_model_name = Config().trainer.global_model_name
            prefix_names = self.global_model_name.split("__")
            logging.info(
                f"Extracting the global parameters with prefix {prefix_names}."
            )

        if model is None:
            if self.global_model_name == "whole":
                return self.model.cpu().state_dict()
            else:
                full_weights = self.model.cpu().state_dict()
                extracted_weights = OrderedDict([
                    (name, param) for name, param in full_weights.items()
                    if is_global_parameter(name, self.global_model_name)
                ])
                return extracted_weights

        else:
            return model.cpu().state_dict()

    def is_incomplete_weights(self, weights):
        model_params = self.model.state_dict()
        for para_name in list(model_params.keys()):
            if para_name not in weights:
                return True

        return False

    def complete_weights(self, weights, pool_weights):
        """ Completet the weights based on the pool_weights. """
        model_params = self.model.state_dict()
        completed_weights = OrderedDict()
        for para_name in list(model_params.keys()):
            if para_name in weights:
                completed_weights[para_name] = weights[para_name]
            else:
                completed_weights[para_name] = pool_weights[para_name]

        return completed_weights

    def load_weights(self, weights, strict=False):
        """Load the model weights passed in as a parameter.

            In the client side, we should complete the weights (OrderDict) from the
            server if the weights only contain part of the whole model.
            Thus, the 'strict' should be True, as the completed weights should
            contain same paras as the client's model.
            In the server side, there is no need to complete the weights as the
            server will always extract weights by self.extract_weights to obtain the
            desired weights, i.e., ensure always operate on the global model.
            Thus, the 'strict' should be False as we only load the operated weights to
            the model.

            But wihout losing generality, we set strict = False here as the algorithm class
            is utilized by the server and the client simutaneously.
        """
        self.model.load_state_dict(weights, strict=strict)
