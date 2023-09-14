import os
import logging
from collections import OrderedDict

from plato.algorithms import fedavg
from plato.config import Config


class Algorithm(fedavg.Algorithm):
    """The federated learning algorithm for MistNetPlus, used by the server."""

    def update_weights(self, deltas):
        """Aggregates the weights received into baseline weights."""
        baseline_weights = self.extract_weights()

        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            if name == Config().parameters.model.cut_layer:
                logging.info("[Server #%d] %s cut", os.getpid(), name)
                break

            updated_weights[name] = weight + deltas[name]

        return updated_weights
