"""
The federated averaging algorithm for MindSpore.
"""
from collections import OrderedDict

from plato.algorithms import base
from plato.datasources import registry as datasources_registry


class Algorithm(base.Algorithm):
    """TensorFlow-based federated averaging algorithm, used by both the client and the server."""
    def extract_weights(self):
        """ Extract weights from the model. """
        #TensorFlow needs to build the model first using the input shape from the dataset
        datasource = datasources_registry.get()
        self.model.build_model(datasource.input_shape())
        weights = self.model.get_weights()
        return weights

    def compute_weight_updates(self, weights_received):
        """ Extract the weights received from a client and compute the updates. """
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
        datasource = datasources_registry.get()
        self.model.build_model(datasource.input_shape())

        self.model.set_weights(weights)