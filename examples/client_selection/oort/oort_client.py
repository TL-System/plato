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

        num_samples = getattr(report, "num_samples", 0)
        if num_samples > 0 and train_squared_loss_step:
            mean_squared_loss = sum(train_squared_loss_step) / num_samples
            report.statistical_utility = num_samples * np.sqrt(mean_squared_loss)
        else:
            report.statistical_utility = 0.0

        return report
