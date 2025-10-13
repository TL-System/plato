"""
A personalized federated learning trainer with Ditto.

Reference:
Li, T., Hu, S., Beirami, A., & Smith, V. (2021).
"Ditto: Fair and Robust Federated Learning Through Personalization."
In Proceedings of ICML 2021.

Paper: https://arxiv.org/abs/2012.04221
"""

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import DittoUpdateStrategyFromConfig


class Trainer(ComposableTrainer):
    """
    A trainer with Ditto using composition-based design.

    Ditto is a personalized federated learning algorithm that maintains two models:
    1. Global model (w): Trained via standard federated learning
    2. Personalized model (v): Trained locally with regularization towards global model

    The personalized model training happens after the global model training completes.
    The personalized model optimizes:
        min_v F_k(v) + λ/2 * ||v - w||^2

    where:
    - F_k(v) is the local loss on client k's data
    - w is the global model
    - λ controls the strength of regularization towards the global model

    Training procedure:
    1. Train global model (w) with standard federated learning
    2. After global training, train personalized model (v) with regularization
    3. Send only global model to server; keep personalized model locally
    4. Use personalized model for inference in final personalization rounds

    The lambda coefficient and personalization epochs are read from the configuration file.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the Ditto trainer with composition-based strategies.

        Args:
            model: The neural network model to train (used as global model template)
            callbacks: Optional list of callback handlers
        """
        super().__init__(
            model=model,
            callbacks=callbacks,
            model_update_strategy=DittoUpdateStrategyFromConfig(model_fn=None),
        )
