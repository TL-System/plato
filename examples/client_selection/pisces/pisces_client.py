"""
Pisces: an asynchronous client selection and server aggregation algorithm.

Reference:

Z. Jiang, B. Wang, B. Li, B. Li. "Pisces: Efficient Federated Learning via Guided Asynchronous
Training," in Proceedings of ACM Symposium on Cloud Computing (SoCC), 2022.

URL: https://arxiv.org/abs/2206.09264
"""

import math
from types import SimpleNamespace

from plato.clients import simple


class Client(simple.Client):
    """
    A Pisces federated learning client who sends weight updates and client statistical utility.
    """

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.loss_decay = 1e-2

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Compute the moving average of batch loss for statistical utility."""
        train_batch_loss = [
            float(loss)
            for loss in self.trainer.run_history.get_metric_values("train_batch_loss")
        ]

        moving_average_sq_loss = None

        for batch_loss in train_batch_loss:
            squared_loss = batch_loss**2
            if moving_average_sq_loss is None:
                moving_average_sq_loss = squared_loss
            else:
                moving_average_sq_loss = (
                    1 - self.loss_decay
                ) * moving_average_sq_loss + self.loss_decay * squared_loss

        if moving_average_sq_loss is not None and report.num_samples > 0:
            loss_norm = math.sqrt(moving_average_sq_loss)
            report.statistical_utility = report.num_samples * loss_norm
            report.moving_loss_norm = loss_norm
        else:
            report.statistical_utility = 0.0
            report.moving_loss_norm = 0.0

        report.start_round = self.current_round
        return report
