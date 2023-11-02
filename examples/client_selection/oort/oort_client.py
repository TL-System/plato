"""
A federated learning client using Oort.

Reference:

F. Lai, X. Zhu, H. V. Madhyastha and M. Chowdhury, "Oort: Efficient Federated Learning via
Guided Participant Selection," in USENIX Symposium on Operating Systems Design and Implementation
(OSDI 2021), July 2021.
"""

from types import SimpleNamespace

import numpy as np
from plato.clients import simple


class Client(simple.Client):
    """
    A federated learning client that calculates its statistical utility
    """

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Wrap up generating the report with any additional information."""
        train_squared_loss_step = self.trainer.run_history.get_metric_values(
            "train_squared_loss_step"
        )

        report.statistical_utility = report.num_samples * np.sqrt(
            1.0 / report.num_samples * sum(train_squared_loss_step)
        )

        return report
