"""FedProx trainer preserving Plato's legacy unsquared proximal objective."""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms.fedprox_strategy import (
    FedProxLossStrategyFromConfig,
)


class Trainer(ComposableTrainer):
    """Apply the configured proximal loss with an ordinary optimizer."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=FedProxLossStrategyFromConfig(),
        )
