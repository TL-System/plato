"""
The federated averaging algorithm for self-supervised method.

The main target of this algorithm is to achieve the property of
randomly exchanging the whole defined or its sub-module between
the server and clients.
Thus, the 'extract_weights' functions extract the model' parameters
based on the required 'global_model_name'.

"""
from collections import OrderedDict

from plato.algorithms import fedavg
from plato.config import Config
from plato.trainers.base import Trainer


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

        if model is None:
            if self.global_model_name == "whole":
                return self.model.cpu().state_dict()
            else:
                full_weights = self.model.cpu().state_dict()
                extracted_weights = OrderedDict([
                    (name, param) for name, param in full_weights.items()
                    if self.global_model_name in name
                ])
                return extracted_weights

        else:
            return model.cpu().state_dict()

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=False)
