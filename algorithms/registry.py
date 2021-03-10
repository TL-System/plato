"""
The registry for algorithms that contains framework-specific implementations.

Having a registry of all available classes is convenient for retrieving an instance
based using a configuration at run-time.
"""
import logging
from collections import OrderedDict

from algorithms import (
    fedavg,
    mistnet,
)

from config import Config

registered_algorithms = OrderedDict([
    ('fedavg', fedavg.Algorithm),
    ('mistnet', mistnet.Algorithm),
])


def get(model, trainer, client_id=None):
    """Get the algorithm with the provided name."""
    algorithm_name = Config().algorithm.type

    # There is no frameworks-specific algorithm by default
    registered_alg = None

    if algorithm_name in registered_algorithms:
        logging.info("Algorithm: %s", algorithm_name)
        registered_alg = registered_algorithms[algorithm_name](model, trainer,
                                                               client_id)

    return registered_alg
