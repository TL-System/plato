"""
A federated learning client that sends its statistical utility
"""

from types import SimpleNamespace

import numpy as np
from plato.clients import simple


class Client(simple.Client):
    """
    A federated learning client that calculates its statistical utility
    """

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.statistical_utility = None

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Wrap up generating the report with any additional information."""
        sum_loss = self.trainer.run_history.get_latest_metric("train_squared_loss_sum")

        self.statistical_utility = np.abs(report.num_samples) * np.sqrt(
            1.0 / report.num_samples * sum_loss
        )
        report.statistics_utility = self.statistical_utility
        return report
