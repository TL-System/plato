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
    adaptive_sync,
    adaptive_freezing,
)

from config import Config

registered_algorithms = OrderedDict([
    ('fedavg', fedavg.Algorithm),
    ('mistnet', mistnet.Algorithm),
    ('adaptive_sync', adaptive_sync.Algorithm),
    ('adaptive_freezing', adaptive_freezing.Algorithm),
])

if hasattr(Config().trainer, 'use_mindspore'):
    from algorithms.mindspore import (
        fedavg as fedavg_mindspore,
        mistnet as mistnet_mindspore,
    )
    registered_algorithms += OrderedDict([
        ('fedavg_mindspore', fedavg_mindspore.Algorithm),
        ('mistnet_mindspore', mistnet_mindspore.Algorithm),
    ])


def get(model, trainer=None, client_id=None):
    """Get the algorithm with the provided name."""
    algorithm_name = Config().algorithm.type

    # There is no frameworks-specific algorithm by default
    registered_alg = None

    if algorithm_name in registered_algorithms:
        logging.info("Algorithm: %s", algorithm_name)
        registered_alg = registered_algorithms[algorithm_name](model, trainer,
                                                               client_id)

    return registered_alg
