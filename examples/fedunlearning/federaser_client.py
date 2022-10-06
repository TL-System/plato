"""
A customized client for federaser

Liu et al., "FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning
Models," in 2021 IEEE/ACM 29th International Symposium on Quality of Service (IWQoS 2021).

Reference: https://ieeexplore.ieee.org/document/9521274
"""

from tkinter import N
import numpy as np
from plato.config import Config
import fedunlearning_client


class Client(fedunlearning_client.Client):
    """A federated unlearning client that implements the FedEraser Algorithm.

    The clients processe server's retraining response and set trainer to retraining phase.
    """

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model=model,
                         datasource=datasource,
                         algorithm=algorithm,
                         trainer=trainer)

    def process_server_response(self, server_response):
        """
        If the retraining starts, set the trainer to retraining phase
        """
        updated_epochs = server_response['updated_epochs']
        if updated_epochs is not None:
            self.trainer.set_epochs(updated_epochs)
        else:
            self.trainer.set_epochs(Config().trainer.epochs)

