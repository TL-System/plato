"""
An implementation of the FedDyn algorithm.

D. Acar, et al., "Federated Learning Based on Dynamic Regularization," in the
Proceedings of ICLR 2021.

Paper: https://openreview.net/forum?id=B7v4QMR6Z9w

Source code: https://github.com/alpemreacar/FedDyn
"""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    FedDynLossStrategyFromConfig,
    FedDynUpdateStrategy,
)


class Trainer(ComposableTrainer):
    """
    FedDyn's Trainer with composition-based design.

    FedDyn uses dynamic regularization to address client drift. The local
    objective includes:
    1. Standard task loss
    2. Linear penalty term: -<w, h_k> where h_k = w_prev - w_global
    3. L2 regularization: (α/2)||w - w^t||^2

    The alpha coefficient is read from the configuration file and can be
    adaptively scaled by client data weight.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the FedDyn trainer with composition-based strategies.

        Args:
            model: The neural network model to train
            callbacks: Optional list of callback handlers
        """
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=FedDynLossStrategyFromConfig(),
            model_update_strategy=FedDynUpdateStrategy(),
        )
