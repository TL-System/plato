"""
A personalized federated learning training algorithm using FedRep.

Reference:

Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", in the Proceedings of ICML 2021.

https://arxiv.org/abs/2102.07078

Source code: https://github.com/lgcollins/FedRep
"""
from collections import OrderedDict

from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """ The federated learning algorithm for FedRep, used by the server. """

    def __init__(self, trainer=None):
        super().__init__(trainer)

        self.representation_param_names = []

    def set_global_parameter_names(self, global_parameter_names):
        """ Setting the global representation weights. """
        self.representation_param_names = global_parameter_names

    def extract_weights(self, model=None):
        """ Extract weights from the model. """

        def extract_required_weights(model, parameter_names):
            full_weights = model.state_dict()

            extracted_weights = OrderedDict([
                (name, param) for name, param in full_weights.items()
                if name in parameter_names
            ])
            return extracted_weights

        if model is None:
            return extract_required_weights(self.model.cpu(),
                                            self.representation_param_names)

        return extract_required_weights(model.cpu(),
                                        self.representation_param_names)

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=False)
