"""
The enhanced federated averaging algorithm for PyTorch.

It adds more properties to the original implementation of fedavg.py.

These properties are:

    - support: (extract_weights)
                Randomly set one part of the whole model as the global model
                that will be exchanged between clients and the server.
               This can be defined by the hyper-parameter in the config file
               'global_model_name: xxx'.

               For example, when the whole model is the combination of
                encoder + classifier
                the global model can be set to be:
                * only the encoder as the global model
                    global_model_name: encoder
                * only the classifier as the global model
                    global_model_name: classifier
                * equalivent to the whole model as the global model
                    global_model_name: encoder__classifier
                 or global_model_name: whole

    - support: (extract_parameters_prefix)
                Extract the prefixes of the parameters in one model.
                For example, if the model is the combination of encoder and classifier,
                i.e., encoder + classifier.
                The parameters of the encoder will be added prefix 'encoder.'
                while the prefix for parameters of classifier is 'classifier'.
                With this function, we can extract these unique prefixes.
    - support: (complete_weights)
                Complete the given weights to make the completed weights have same
                parameters as the self.model.


In general, this enhanced fedavg is generally utilized in personalized federated learning.
The main reason is that each client can hold models on its side to complete the downloaded global
model.

This is why to denote this implementation to be 'fedavg_pers.py'

This fedavg_pers is commonly utilized in cooperation with the clients/pers_simple.py

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
            - online_network__online_predictor
            - encoder
            - encoder__cls_fc
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

    def extract_parameters_prefix(self, parameters_name):
        """ Extracting the major prefixs of a group of parameters. """
        extracted_prefix = []

        # the para name is presented as encoder.xxx.xxx
        # that is combined by the key_word "."
        # we aim to extract the encoder
        split_str = "."
        for para_name in parameters_name:
            splitted_names = para_name.split(split_str)
            prefix = splitted_names[0]
            # add the obtained prefix to the list, if
            #   - empty list
            #   - a new prefix
            # add to the empty list directly
            if prefix not in extracted_prefix:
                extracted_prefix.append(prefix)

        return extracted_prefix

    def complete_weights(self, weights, pool_weights):
        """ Completet the weights for self.model based on the
            'weights' and 'pool_weights'.

            The 'weights' has the highes pariority, thus the para data
            in 'weights' will be used for self.model's completation.
            Then, if 'weights' does not contain the parameters of self.model,
            the weights in 'pool_weights' will be utilized.

            Therefore, the 'pool_weights' is regarded as a parameters pool for
            others to select from.

            Args:
                weights (OrderDict): the obtained OrderDict containing
                    parameters that are generally extracted by state_dict
                pool_weights (OrderDict): the same structure and data type
                    as 'weights'.

            Returns:
                completed_weights (OrderDict): contains the full parameters
                    as self.model, thus can be directly assigned to the
                    self.model
                prefixes_of_existence (list): the prefixes for parameters
                    in 'weights'
                prefixes_of_completion (list): the prefixes for parameters
                    that are selected from the 'pool_weights' to be used
                    for completion.
        """
        weights_of_completion = list()
        weights_of_existence = list()
        model_params = self.model.state_dict()
        completed_weights = OrderedDict()
        for para_name in list(model_params.keys()):
            if para_name in weights:
                completed_weights[para_name] = weights[para_name]
                weights_of_existence.append(para_name)
            else:
                completed_weights[para_name] = pool_weights[para_name]
                weights_of_completion.append(para_name)

        prefixes_of_existence = self.extract_parameters_prefix(
            weights_of_existence)
        prefixes_of_completion = self.extract_parameters_prefix(
            weights_of_completion)

        return completed_weights, prefixes_of_existence, prefixes_of_completion

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
