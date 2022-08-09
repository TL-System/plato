"""
A federated learning client using pruning in FedSCR.
"""

import logging
import pickle
from dataclasses import dataclass

from plato.config import Config
from plato.clients import simple


@dataclass
class FedSCRReport(simple.Report):
    """A client report containing the final loss, to be sent to the FedSCR server for the adaptive
    algorithm."""

    loss: float
    div: float
    avg_update: float


class Client(simple.Client):
    """
    A federated learning client prunes its update before sending out.
    """

    def client_train_end(self):
        """Method called at the end of local training."""

        logging.info("[Client #%d] Trained with FedSCR algorithm.", self.client_id)

        if self.trainer.use_adaptive:
            final_loss = self.get_loss()
            divs = self.get_divs()
            self.report = FedSCRReport(
                self.report.num_samples,
                self.report.accuracy,
                self.report.training_time,
                self.report.comm_time,
                self.report.update_response,
                final_loss,
                divs["div"],
                divs["g"],
            )

    def get_loss(self):
        """Retrieve the loss value from the training process."""
        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}.loss"
        loss = self.trainer._load_loss(filename)
        return loss

    def get_divs(self):
        """Retrieve the average weight update and divergences."""
        model_path = Config().params["checkpoint_path"]
        model_name = Config().trainer.model_name

        report_path = f"{model_path}/{model_name}_{self.client_id}.pkl"

        with open(report_path, "rb") as file:
            divs = pickle.load(file)

        return divs
