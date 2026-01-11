"""
A FedALA trainer built with the composable strategy architecture.
"""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedALAUpdateStrategyFromConfig


class Trainer(ComposableTrainer):
    """
    The federated learning trainer for FedALA clients.

    This trainer applies adaptive local aggregation to initialize the local
    model before training, using hyperparameters from the configuration file.
    """

    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            model_update_strategy=FedALAUpdateStrategyFromConfig(),
        )
