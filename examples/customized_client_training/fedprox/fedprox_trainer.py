"""
A federated learning training session using FedProx.

To better handle system heterogeneity, the FedProx algorithm introduced a
proximal term in the optimizer used by local training on the clients. It has
been quite widely cited and compared with in the federated learning literature.

Reference:
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
"Federated optimization in heterogeneous networks." Proceedings of Machine
Learning and Systems (MLSys), vol. 2, 429-450.

https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf
"""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedProxLossStrategyFromConfig


class Trainer(ComposableTrainer):
    """
    The federated learning trainer for the FedProx client.

    This trainer uses the composition-based design with FedProx loss strategy.
    The proximal term coefficient (mu) is read from the configuration file.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the FedProx trainer with composition-based strategies.

        Args:
            model: The neural network model to train
            callbacks: Optional list of callback handlers
        """
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=FedProxLossStrategyFromConfig(),
        )
