"""
The registry for algorithms that contains framework-specific implementations.

Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
import logging
from collections import OrderedDict

from plato.config import Config

if hasattr(Config().trainer, 'use_mindspore'):
    from algorithms.mindspore import (
        fedavg as fedavg_mindspore,
        mistnet as mistnet_mindspore,
    )

    registered_algorithms = OrderedDict([
        ('fedavg', fedavg_mindspore.Algorithm),
        ('mistnet', mistnet_mindspore.Algorithm),
    ])
else:
    from plato.algorithms import (
        fedavg,
        mistnet,
        adaptive_sync,
        adaptive_freezing,
        split_learning,
    )

    registered_algorithms = OrderedDict([
        ('fedavg', fedavg.Algorithm),
        ('mistnet', mistnet.Algorithm),
        ('adaptive_sync', adaptive_sync.Algorithm),
        ('adaptive_freezing', adaptive_freezing.Algorithm),
        ('split_learning', split_learning.Algorithm),
    ])


def get(trainer=None, client_id=None):
    """Get the algorithm with the provided type."""
    algorithm_type = Config().algorithm.type

    if algorithm_type in registered_algorithms:
        logging.info("Algorithm: %s", algorithm_type)
        registered_alg = registered_algorithms[algorithm_type](trainer,
                                                               client_id)
        return registered_alg
    else:
        raise ValueError('No such model: {}'.format(algorithm_type))
