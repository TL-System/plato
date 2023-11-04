"""
A federated learning client of FedSCR.
"""

import logging
from types import SimpleNamespace

from plato.clients import simple


class Client(simple.Client):
    """
    A federated learning client prunes its update before sending out.
    """

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if self.trainer.use_adaptive:
            if "update_thresholds" in server_response:
                # Load its update threshold
                self.trainer.update_threshold = server_response["update_thresholds"][
                    str(self.client_id)
                ]
                logging.info(
                    "[Client #%d] Received update threshold %.2f",
                    self.client_id,
                    self.trainer.update_threshold,
                )

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Wraps up generating the report with any additional information."""
        if self.trainer.use_adaptive:
            report.div_from_global = self.trainer.run_history.get_latest_metric(
                "div_from_global"
            )
            report.avg_update = self.trainer.run_history.get_latest_metric("avg_update")
            report.loss = self.trainer.run_history.get_latest_metric("train_loss")

        return report
