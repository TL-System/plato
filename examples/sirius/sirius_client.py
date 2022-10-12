"""
An asynchronous federated learning client using Sirius.
"""
from types import SimpleNamespace
import numpy as np
from plato.clients import simple


class Client(simple.Client):
    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Wrap up generating the report with any additional information."""

        train_squared_loss_step = self.trainer.run_history.get_latest_metric(
            "train_loss"
        )
        report.statistics_utility = report.num_samples * np.sqrt(
            1.0 / report.num_samples * train_squared_loss_step
        )
        report.start_round = self.current_round
        return report
