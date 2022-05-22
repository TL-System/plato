""" Implement the aggregation algorithm for FedRep. 

"""
from collections import OrderedDict

from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """The federated learning trainer for Adaptive Synchronization Frequency,
       used by the server.
    """

    def __init__(self, trainer=None):
        super().__init__(trainer)

        self.representation_weights_key = []

    def set_global_weights_key(self, global_keys):
        """ Setting the global representation weights. """
        self.representation_weights_key = global_keys

    def extract_weights(self, model=None):
        """ Extract weights from the model. """

        def extract_required_weights(model, weights_key):
            full_weights = model.state_dict()

            extracted_weights = OrderedDict([
                (name, param) for name, param in full_weights.items()
                if name in weights_key
            ])
            return extracted_weights

        if model is None:
            return extract_required_weights(self.model.cpu(),
                                            self.representation_weights_key)
        else:
            return extract_required_weights(model.cpu(),
                                            self.representation_weights_key)

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=False)
