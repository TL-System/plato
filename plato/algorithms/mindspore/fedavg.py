"""
The federated averaging algorithm for MindSpore.
"""
from collections import OrderedDict
import numpy as np

import mindspore

from plato.algorithms import base


class Algorithm(base.Algorithm):
    """MindSpore-based federated averaging algorithm, used by both the client and the server."""

    def extract_weights(self, model=None):
        """Extract weights from the model."""
        numpy_weights = OrderedDict()
        if model is None:
            for name, weight in self.model.parameters_dict().items():
                numpy_weights[name] = weight.asnumpy()
        else:
            for name, weight in model.parameters_dict().items():
                numpy_weights[name] = weight.asnumpy()

        return numpy_weights

    def compute_weight_deltas(self, baseline_weights, weights_received):
        """Compute the deltas between baseline weights and weights received."""
        # Calculate updates from the received weights
        deltas = []
        for weight in weights_received:
            delta = OrderedDict()
            for name, current_weight in weight.items():
                baseline = baseline_weights[name]

                # Calculate update
                _delta = current_weight - baseline
                delta[name] = _delta
            deltas.append(delta)

        return deltas

    def update_weights(self, deltas):
        """Updates the existing model weights from the provided deltas."""
        baseline_weights = self.extract_weights()

        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + deltas[name]

        return updated_weights

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        for name, weight in weights.items():
            weight_tensor = mindspore.Tensor(weight.astype(np.float32))
            weights[name] = mindspore.Parameter(weight_tensor, name=name)

        # One can also use `self.model.load_parameter_slice(weights)', which
        # seems to be equivalent to mindspore.load_param_into_net() in its effects

        mindspore.load_param_into_net(self.model, weights, strict_load=True)

    def print_weights(self):
        """Print all the weights from the model."""
        for _, param in self.model.parameters_and_names():
            print(f"key = {param.name}, value = {param.asnumpy()}")
