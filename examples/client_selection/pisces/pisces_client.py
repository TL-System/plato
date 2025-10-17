"""
Pisces: an asynchronous client selection and server aggregation algorithm.

Reference:

Z. Jiang, B. Wang, B. Li, B. Li. "Pisces: Efficient Federated Learning via Guided Asynchronous
Training," in Proceedings of ACM Symposium on Cloud Computing (SoCC), 2022.

URL: https://arxiv.org/abs/2206.09264
"""

from plato.clients import simple
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultReportingStrategy


class PiscesReportingStrategy(DefaultReportingStrategy):
    """Reporting strategy that tracks Pisces statistical utility."""

    def __init__(self, loss_decay: float) -> None:
        super().__init__()
        self.loss_decay = loss_decay

    def build_report(self, context: ClientContext, report):
        report = super().build_report(context, report)

        trainer = context.trainer
        if trainer is None or getattr(trainer, "run_history", None) is None:
            report.statistical_utility = 0.0
            report.moving_loss_norm = 0.0
            report.start_round = context.current_round
            return report

        train_batch_loss = [
            float(loss)
            for loss in trainer.run_history.get_metric_values("train_batch_loss")
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

        num_samples = getattr(report, "num_samples", 0)
        if moving_average_sq_loss is not None and num_samples > 0:
            loss_norm = moving_average_sq_loss**0.5
            report.statistical_utility = num_samples * loss_norm
            report.moving_loss_norm = loss_norm
        else:
            report.statistical_utility = 0.0
            report.moving_loss_norm = 0.0

        report.start_round = context.current_round
        return report


class Client(simple.Client):
    """
    A Pisces federated learning client who sends weight updates and client statistical utility.
    """

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
        *,
        loss_decay: float = 1e-2,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=trainer_callbacks,
        )

        self._configure_composable(
            lifecycle_strategy=self.lifecycle_strategy,
            payload_strategy=self.payload_strategy,
            training_strategy=self.training_strategy,
            reporting_strategy=PiscesReportingStrategy(loss_decay),
            communication_strategy=self.communication_strategy,
        )
