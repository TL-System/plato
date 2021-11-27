"""
The federated averaging algorithm for MindSpore.
"""
from collections import OrderedDict
import numpy as np

import mindspore

from plato.algorithms import base


class Algorithm(base.Algorithm):
    """MindSpore-based federated averaging algorithm, used by both the client and the server."""
    def extract_weights(self):
        """Extract weights from the model."""

        # It turns out that recent MindSpore releases later than 1.1.3, including 1.5.0, do not support
        # `pickle.load()` on `Tensor` objects (https://gitee.com/mindspore/mindspore/issues/I43RPP?from=project-issue).
        # Therefore, Tensor objects must be converted to numpy arrays first before being pickled.

        numpy_weights = OrderedDict()

        for name, weight in self.model.parameters_dict().items():
            numpy_weights[name] = weight.asnumpy()

        return numpy_weights

    def print_weights(self):
        """Print all the weights from the model."""
        for _, param in self.model.parameters_and_names():
            print(f'key = {param.name}, value = {param.asnumpy()}')

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        for name, weight in weights.items():
            weight_tensor = mindspore.Tensor(weight.astype(np.float32))
            weights[name] = mindspore.Parameter(weight_tensor, name=name)

        # One can also use `self.model.load_parameter_slice(weights)', which
        # seems to be equivalent to mindspore.load_param_into_net() in its effects

        mindspore.load_param_into_net(self.model, weights, strict_load=True)
