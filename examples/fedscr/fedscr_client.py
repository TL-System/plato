"""
A federated learning client of FedSCR.
"""

import logging
import pickle
from types import SimpleNamespace

from plato.clients import simple
from plato.config import Config


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
        """Wrap up generating the report with any additional information."""

        if self.trainer.use_adaptive:
            additional_info = self.get_additional_info()
            report.div_from_global = additional_info["div_from_global"]
            report.avg_update = additional_info["avg_update"]
            report.loss = additional_info["final_loss"]

        return report

    # pylint: disable=protected-access
    def get_additional_info(self):
        """Retrieve the average weight update and weight divergence."""
        model_path = Config().params["checkpoint_path"]
        model_name = Config().trainer.model_name

        report_path = f"{model_path}/{model_name}_{self.client_id}.pkl"

        with open(report_path, "rb") as file:
            additional_info = pickle.load(file)

        return additional_info
