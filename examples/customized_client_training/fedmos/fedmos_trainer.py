"""
An implementation of the FedMos algorithm.

X. Wang, Y. Chen, Y. Li, X. Liao, H. Jin and B. Li, "FedMoS: Taming Client Drift in Federated Learning with Double Momentum and Adaptive Selection," IEEE INFOCOM 2023

Paper: https://ieeexplore.ieee.org/document/10228957

Source code: https://github.com/Distributed-Learning-Networking-Group/FedMoS
"""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    FedMosOptimizerStrategyFromConfig,
    FedMosStepStrategy,
    FedMosUpdateStrategy,
)


class Trainer(ComposableTrainer):
    """
    FedMos's Trainer with composition-based design.

    FedMos uses double momentum to address client drift:
    1. Local momentum: Standard momentum in the optimizer
    2. Global momentum: Momentum towards the global model

    The optimizer implements: w = (1-mu)*w - lr*m + mu*w_global

    The momentum coefficients (a for local, mu for global) are read from
    the configuration file.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the FedMos trainer with composition-based strategies.

        Args:
            model: The neural network model to train
            callbacks: Optional list of callback handlers
        """
        super().__init__(
            model=model,
            callbacks=callbacks,
            optimizer_strategy=FedMosOptimizerStrategyFromConfig(),
            model_update_strategy=FedMosUpdateStrategy(),
            training_step_strategy=FedMosStepStrategy(),
        )
