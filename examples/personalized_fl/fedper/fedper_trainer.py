"""
A personalized federated learning trainer with FedPer.

Reference:
Arivazhagan, M. G., Aggarwal, V., Singh, A. K., & Choudhary, S. (2019).
"Federated Learning with Personalization Layers."
arXiv preprint arXiv:1912.00818.

Paper: https://arxiv.org/abs/1912.00818
"""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedPerUpdateStrategyFromConfig


class Trainer(ComposableTrainer):
    """
    A trainer with FedPer, which freezes the global model layers in the final
    personalization round.

    FedPer maintains global (shared) and local (personalized) layers. During
    regular federated learning rounds, all layers are trained. During final
    personalization rounds (after Config().trainer.rounds), global layers are
    frozen and only local layers are trained.

    The layer names are read from the configuration file.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the FedPer trainer with composition-based strategies.

        Args:
            model: The neural network model to train
            callbacks: Optional list of callback handlers
        """
        super().__init__(
            model=model,
            callbacks=callbacks,
            model_update_strategy=FedPerUpdateStrategyFromConfig(),
        )
