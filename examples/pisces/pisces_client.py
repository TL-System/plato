"""
Pisces: an asynchronous client selection and server aggregation algorithm.

Reference:

Z. Jiang, B. Wang, B. Li, B. Li. "Pisces: Efficient Federated Learning via Guided Asynchronous
Training," in Proceedings of ACM Symposium on Cloud Computing (SoCC), 2022.

URL: https://arxiv.org/abs/2206.09264
"""
from types import SimpleNamespace
import numpy as np
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
        train_batch_loss = self.trainer.run_history.get_metric_values(
            "train_batch_loss"
        )

        moving_average_loss = 0

        for batch_loss in train_batch_loss:
            moving_average_loss = (
                1 - self.loss_decay
            ) * moving_average_loss + self.loss_decay * batch_loss

        train_squared_loss = np.sqrt(moving_average_loss.item())

        report.statistical_utility = report.num_samples * np.sqrt(
            1.0 / report.num_samples * train_squared_loss
        )
        report.start_round = self.current_round
        return report
