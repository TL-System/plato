"""
The federated averaging algorithm for MindSpore.
"""
from collections import OrderedDict

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

    def compute_weight_updates(self, weights_received):
        """Extract the weights received from a client and compute the updates."""
        # Extract baseline model weights
        baseline_weights = self.extract_weights()

        # Calculate updates from the received weights
        updates = []
        for weight in weights_received:
            update = OrderedDict()
            for name, current_weight in weight.items():
                baseline = baseline_weights[name]

                # Calculate update
                delta = current_weight - baseline
                update[name] = delta
            updates.append(update)

        return updates

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        for name, weight in weights.items():
            weights[name] = mindspore.Parameter(weight, name=name)

        # One can also use `self.model.load_parameter_slice(weights)', which
        # seems to be equivalent to mindspore.load_param_into_net() in its effects

        mindspore.load_param_into_net(self.model, weights, strict_load=True)
