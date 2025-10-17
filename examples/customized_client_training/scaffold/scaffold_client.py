"""
A federated learning client using SCAFFOLD.

Reference:

Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,"
in Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

https://arxiv.org/pdf/1910.06378.pdf
"""

import logging
import os
import pickle

from plato.clients import simple
from plato.clients.strategies.defaults import DefaultLifecycleStrategy
from plato.config import Config


class ScaffoldLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that initialises SCAFFOLD control variates."""

    def configure(self, context) -> None:
        super().configure(context)

        trainer = context.trainer
        if trainer is None:
            return

        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name
        filename = f"{model_name}_{context.client_id}_control_variate.pth"
        client_control_variate_path = f"{model_path}/{filename}"

        if os.path.exists(client_control_variate_path):
            logging.info(
                "[Client #%d] Loading the control variate from %s.",
                context.client_id,
                client_control_variate_path,
            )
            with open(client_control_variate_path, "rb") as path:
                client_control_variate = pickle.load(path)
            trainer.client_control_variate = client_control_variate
            context.state["client_control_variate"] = client_control_variate
            if context.owner is not None:
                context.owner.client_control_variate = client_control_variate
        else:
            trainer.client_control_variate = None
            if context.owner is not None:
                context.owner.client_control_variate = None

        trainer.client_control_variate_path = client_control_variate_path


class Client(simple.Client):
    """A SCAFFOLD federated learning client who sends weight updates
    and client control variate."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        self.client_control_variate = None

        self._configure_composable(
            lifecycle_strategy=ScaffoldLifecycleStrategy(),
            payload_strategy=self.payload_strategy,
            training_strategy=self.training_strategy,
            reporting_strategy=self.reporting_strategy,
            communication_strategy=self.communication_strategy,
        )
