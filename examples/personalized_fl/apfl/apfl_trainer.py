"""
A personalized federated learning trainer using APFL.

Reference:
Deng, Y., Kamani, M. M., & Mahdavi, M. (2020).
"Adaptive Personalized Federated Learning."
arXiv preprint arXiv:2003.13461.

Paper: https://arxiv.org/abs/2003.13461
"""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    APFLStepStrategy,
    APFLUpdateStrategyFromConfig,
)


class Trainer(ComposableTrainer):
    """
    A trainer using the APFL algorithm with composition-based design.

    APFL (Adaptive Personalized Federated Learning) maintains two models:
    1. Global model (w): Received from server and updated via federated averaging
    2. Personalized model (v): Kept locally and optimized for client's data

    The key innovation is an adaptive mixing parameter α that determines the
    interpolation between the two models:
        output = α * v + (1 - α) * w

    The parameter α is learned adaptively for each client based on their local
    data, allowing clients to determine the optimal balance between personalization
    and global knowledge.

    The initial alpha value and learning rate are read from the configuration file.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the APFL trainer with composition-based strategies.

        Args:
            model: The neural network model to train (used as global model template)
            callbacks: Optional list of callback handlers
        """
        super().__init__(
            model=model,
            callbacks=callbacks,
            model_update_strategy=APFLUpdateStrategyFromConfig(model_fn=None),
            training_step_strategy=APFLStepStrategy(),
        )
