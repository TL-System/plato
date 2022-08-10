"""
A federated learning client of FedSCR.
"""

import logging
import pickle
from dataclasses import dataclass

from plato.config import Config
from plato.clients import simple


@dataclass
class Report(simple.Report):
    """A client report containing the final loss, to be sent to the FedSCR server for the adaptive
    algorithm."""

    loss: float
    div_from_global: float
    avg_update: float


class Client(simple.Client):
    """
    A federated learning client prunes its update before sending out.
    """

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
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

    def customize_report(self, report):
        """Wrap up generating the report with any additional information."""
        final_loss = self.get_loss()
        setattr(report, "loss", final_loss)

        if self.trainer.use_adaptive:
            additional_info = self.get_additional_info()
            setattr(report, "div_from_global", additional_info["div_from_global"])
            setattr(report, "avg_update", additional_info["avg_update"])
        else:
            setattr(report, "div_from_global", None)
            setattr(report, "avg_update", None)

        return report

    # pylint: disable=protected-access
    def get_loss(self):
        """Retrieve the loss value from the training process."""
        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}.loss"
        loss = self.trainer._load_loss(filename)
        return loss

    def get_additional_info(self):
        """Retrieve the average weight update and weight divergence."""
        model_path = Config().params["checkpoint_path"]
        model_name = Config().trainer.model_name

        report_path = f"{model_path}/{model_name}_{self.client_id}.pkl"

        with open(report_path, "rb") as file:
            additional_info = pickle.load(file)

        return additional_info
