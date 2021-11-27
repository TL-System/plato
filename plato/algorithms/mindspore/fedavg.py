"""
The federated averaging algorithm for MindSpore.
"""
from typing import OrderedDict
import mindspore

from plato.algorithms import base


class Algorithm(base.Algorithm):
    """MindSpore-based federated averaging algorithm, used by both the client and the server."""
    def extract_weights(self):
        """Extract weights from the model."""
        return self.model.parameters_dict()

    def print_weights(self):
        """Print all the weights from the model."""
        for _, param in self.model.parameters_and_names():
            print(f'key = {param.name}, value = {param.asnumpy()}')

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        for name, weight in weights.items():
            weights[name] = mindspore.Parameter(weight, name=name)

        # One can also use `self.model.load_parameter_slice(weights)', which
        # seems to be equivalent to mindspore.load_param_into_net() in its effects

        mindspore.load_param_into_net(self.model, weights, strict_load=True)

    @staticmethod
    def weights_to_numpy(weights):
        """Converts weights from a model into numpy format."""
        return OrderedDict([(k, v.asnumpy()) for k, v in weights.items()])

    @staticmethod
    def numpy_to_weights(weights):
        """Converts numpy formatted weights into model weight."""
        return OrderedDict([(k, mindspore.Parameter(v, name=k))
                            for k, v in weights.items()])
