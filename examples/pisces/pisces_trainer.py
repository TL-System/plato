"""
Pisces: an asynchronous client selection and server aggregation algorithm.

Reference:

Z. Jiang, B. Wang, B. Li, B. Li. "Pisces: Efficient Federated Learning via Guided Asynchronous Training,"
in Proceedings of ACM Symposium on Cloud Computing (SoCC), 2022.

URL: https://arxiv.org/abs/2206.09264
"""
import numpy as np
from plato.trainers import basic
from plato.trainers import tracking


class LossTracker(tracking.LossTracker):
    def __init__(self):
        super().__init__()
        self.loss_decay = 1e-2

    def reset(self):
        """do not reset this loss tracker."""

    def update(self, loss_batch_value, batch_size=1):
        """Compute moving average of loss."""
        self.total_loss = (
            1.0 - self.loss_decay
        ) * self.total_loss + self.loss_decay * loss_batch_value

    @property
    def average(self):
        """Recording for each epoch"""
        return np.sqrt(self.total_loss.cpu().detach().item())


class Trainer(basic.Trainer):
    """The federated learning trainer for the Pisces client."""

    def __init__(self, model=None):
        """Customize the loss tracker"""
        super().__init__(model)
        self._loss_tracker = LossTracker()
