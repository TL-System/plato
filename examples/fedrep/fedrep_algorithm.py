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

    def extract_weights(self, model=None):
        """Extract weights from the model."""
        def extract_required_weights(model, weights_key):
            full_weights = model.state_dict()
            extracted_weights = OrderedDict([(name, param)
                                             for name, param in full_weights
                                             if name in weights_key])
            return extracted_weights

        if model is None:
            return extract_required_weights(self.model.cpu())
        else:
            return extract_required_weights(model.cpu())
