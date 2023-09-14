from plato.algorithms import fedavg
from collections import OrderedDict

class Algorithm(fedavg.Algorithm):

    def __init__(self, trainer):
        super().__init__(trainer)
    
    def update_weights(self, deltas, lr):
        """ Update the existing model weights. """
        baseline_weights = self.extract_weights()

        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + deltas[name] * lr

        return updated_weights
