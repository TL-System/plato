"""
A trainer with FedRep.

Reference:
Collins, L., Hassani, H., Mokhtari, A., & Shakkottai, S. (2021).
"Exploiting Shared Representations for Personalized Federated Learning."
In Proceedings of the 38th International Conference on Machine Learning (ICML).

Paper: https://arxiv.org/abs/2102.07078
"""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import FedRepUpdateStrategyFromConfig


class Trainer(ComposableTrainer):
    """
    A trainer with FedRep using composition-based design.

    FedRep (Federated Learning with Representation Learning) alternates between
    training local and global layers during regular federated learning rounds:
    - Train local layers for a certain number of epochs (local_epochs)
    - Train global layers for the remaining epochs

    During final personalization rounds (after Config().trainer.rounds), global
    layers are frozen and only local layers are trained.

    The layer names and local_epochs are read from the configuration file.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the FedRep trainer with composition-based strategies.

        Args:
            model: The neural network model to train
            callbacks: Optional list of callback handlers
        """
        super().__init__(
            model=model,
            callbacks=callbacks,
            model_update_strategy=FedRepUpdateStrategyFromConfig(),
        )
