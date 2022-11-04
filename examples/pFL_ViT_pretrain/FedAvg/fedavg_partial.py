"""
The enhanced federated averaging algorithm to aggregate partial sub-modules of one whole model.

Utilization condition:
    In some scenarios, even given one defined model, the users want to utilize the partial 
    sub-modules as the global model in federated learning. Thus, solely these desired 
    sub-modules will be extracted and aggregated during the learning process. 
    Then, noticing that the names of parameters in one sub-module hold consistent prefixes, 
    we propose this piece of code to support the aforementioned feature by setting the 
    hyper-parameter 'global_submodules_prefix' in the config file. 

For example
 A. Given the defined whole model: encoder + head, the hyper-parameter in the config 
    file 'global_submodules_prefix' under the 'trainer' can be set to:
    - whole     : utilizing the whole model as the global model
    - encoder   : utilizing the encoder as the global model
    - head      : utilizing the head as the global model

    Demo:

        trainer:
            global_submodules_prefix: encoder


 B. Given the defined whole model: encoder1 + encoder2 + encoder3 + head1 + head2,
    the hyper-parameter in the config file 'global_submodules_prefix' under the 'trainer' 
    can be set to 
    - whole                 : utilizing the whole model as the global model
    - encoder1__encoder2    : utilizing solely encoder1 and encoder2 as the global model
    - encoder2__head1       : utilizing solely encoder2 and head as the global model
    - encoder2__head1__head2: utilizing solely encoder2, head1 and head2 as the global model

    Demo:

        trainer:
            global_submodules_prefix: encoder1__encoder2
"""

import logging
from collections import OrderedDict

from plato.algorithms import fedavg
from plato.config import Config
from plato.trainers.base import Trainer


def is_global_parameter(param_name, global_submodules_prefix):
    """ Whether the 'param_name' contained in the 'global_submodules_prefix'.

        Args:
            param_name (str): the name str of the parameter
            global_submodules_prefix (str): the str that defines the modules' name
                of the global model. The sub-str of each module is connected 
                by '__'.
                E.g. encoder1__encoder2

    """
    flag = False
    global_params_prefix = global_submodules_prefix.split("__")
    for para_prefix in global_params_prefix:
        if para_prefix in param_name:
            flag = True

    return flag


class Algorithm(fedavg.Algorithm):
    """ Federated averaging algorithm, used by both the client and the server. """

    def __init__(self, trainer: Trainer):
        super().__init__(trainer=trainer)

        # the sub-module's name that is used as the global model
        # shared among clients.
        # by default, the whole model will be used as the global model
        # i.e., global_submodules_prefix = "whole"
        # however, the user can set which component of the model is used
        # as the global model by declaring its prefix in the config file. 

        self.global_submodules_prefix = "whole"


    def extract_weights(self, model=None):
        """ Extract weights from the model.

            To not break the current Plato's code, we follow the principle:
            1.- Once the model is provided, i,e, model != None, its paramters
                will be extracted directly.

            2.- If the model is None, it is required to extract the default
                model, i.e., the self.model of this algo. Currenty, the
                'global_submodules_prefix' defined in the config file will be used
                to extract the desired sub-module from the self.model

        """
        if hasattr(Config().trainer, "global_submodules_prefix"):
            self.global_submodules_prefix = Config().trainer.global_submodules_prefix
            prefix_names = self.global_submodules_prefix.split("__")
            logging.info(
                f"Extracting the global parameters with prefix {prefix_names}."
            )

        if model is None:
            if self.global_submodules_prefix == "whole":
                return self.model.cpu().state_dict()
            else:
                full_weights = self.model.cpu().state_dict()
                extracted_weights = OrderedDict([
                    (name, param) for name, param in full_weights.items()
                    if is_global_parameter(name, self.global_submodules_prefix)
                ])
                return extracted_weights

        else:
            return model.cpu().state_dict()


    def is_incomplete_weights(self, weights):
        """ Whether the given 'weights' does not hold consistent parameters 
            with the self.model """

        model_params = self.model.state_dict()
        for para_name in list(model_params.keys()):
            if para_name not in weights:
                return True

        return False


    def extract_parameters_unique_prefix(self, parameters_name):
        """ Extracting the unique prefixes from given parameters' names. """

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


    def complete_weights(self, weights, auxiliary_weights):
        """ Completeting the weights for self.model based on the
            'weights' and 'auxiliary_weights'.

            'weights' has the [highest priority], so parameter data in 'weights' 
            is assigned to the corresponding parameters of self.model first.

            Then, if 'weights' does not contain the parameters of self.model, 
            the weights in 'auxiliary_weights' will be utilized.


            Args:
                weights (OrderDict): the obtained OrderDict containing
                    parameters that are generally extracted by the func state_dict()
                auxiliary_weights (OrderDict): the same structure and data type
                    as 'weights'.

            Returns:
                completed_weights (OrderDict): contains the full parameters
                    as self.model, thus can be directly assigned to the
                    self.model
                prefixes_of_existence (list): the prefixes for parameters
                    in 'weights'
                prefixes_of_auxiliary (list): the prefixes for parameters
                    that are selected from the 'auxiliary_weights' to be used
                    for completion.
        """

        # weights that are obtained from the 'auxiliary_weights'
        weights_of_auxiliary = list()
        # weights that are obtained from the 'weights'
        weights_of_existence = list()
        # params of the whole model
        model_params = self.model.state_dict()
        # full weights assigned to the model
        completed_weights = OrderedDict()

        for para_name in list(model_params.keys()):
            if para_name in weights:
                completed_weights[para_name] = weights[para_name]
                weights_of_existence.append(para_name)
            else:
                completed_weights[para_name] = auxiliary_weights[para_name]
                weights_of_auxiliary.append(para_name)

        prefixes_of_existence = self.extract_parameters_unique_prefix(
            weights_of_existence)
        prefixes_of_auxiliary = self.extract_parameters_unique_prefix(
            weights_of_auxiliary)

        return completed_weights, prefixes_of_existence, prefixes_of_auxiliary


    def load_weights(self, weights, strict=False):
        """ Load the model weights passed in as a parameter.

        """
        self.model.load_state_dict(weights, strict=strict)
