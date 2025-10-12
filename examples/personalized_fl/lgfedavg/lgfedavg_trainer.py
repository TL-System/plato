"""
A personalized federated learning trainer with LG-FedAvg.

Reference:
Liang, P. P., Liu, T., Ziyin, L., Allen, N. B., Auerbach, R. P., Brent, D.,
... & Morency, L. P. (2020). "Think Locally, Act Globally: Federated Learning
with Local and Global Representations." arXiv preprint arXiv:2001.01523.
"""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import LGFedAvgStepStrategyFromConfig


class Trainer(ComposableTrainer):
    """
    The LG-FedAvg trainer with composition-based design.

    LG-FedAvg performs two forward and backward passes in one iteration:
    1. First freezes global layers and trains local layers
    2. Then freezes local layers and trains global layers

    The layer names are read from the configuration file.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the LG-FedAvg trainer with composition-based strategies.

        Args:
            model: The neural network model to train
            callbacks: Optional list of callback handlers
        """
        super().__init__(
            model=model,
            callbacks=callbacks,
            training_step_strategy=LGFedAvgStepStrategyFromConfig(),
        )
