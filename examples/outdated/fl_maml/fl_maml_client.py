"""
A personalized federated learning client.
"""

import logging
import os
import pickle

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
from plato.config import Config


class FlMamlLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that toggles personalization tests."""

    def process_server_response(self, context, server_response):
        super().process_server_response(context, server_response)
        if "personalization_test" not in server_response:
            return

        owner = context.owner
        if owner is not None:
            owner.do_personalization_test = True


class Client(simple.Client):
    """A federated learning client."""

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
            callbacks=None,
        )
        self.do_personalization_test = False

        payload_strategy = self.payload_strategy
        training_strategy = self.training_strategy
        reporting_strategy = self.reporting_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=FlMamlLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )

    async def _start_training(self):
        """Complete one round of training on this client."""
        self._load_payload(self.server_payload)
        self.server_payload = None

        if self.do_personalization_test:
            # Train a personalized model based on the current meta model and test it
            # This report only contains accuracy of its personalized model
            report = await self.test_personalized_model()
            payload = "personalization_accuracy"
            self.do_personalization_test = False
        else:
            # Regular local training of FL
            report, payload = await self.train()
            if Config().is_edge_server():
                logging.info(
                    "[Server #%d] Model aggregated on edge server (%s).",
                    os.getpid(),
                    self,
                )
            else:
                logging.info("[%s] Model trained.", self)

        # Sending the client report as metadata to the server (payload to follow)
        await self.sio.emit(
            "client_report", {"id": self.client_id, "report": pickle.dumps(report)}
        )

        # Sending the client training payload to the server
        await self.send(payload)

    async def test_personalized_model(self):
        """A client first trains its personalized model based on
        the global meta model and then test it.
        """
        logging.info("[%s] Started training a personalized model.", self)

        # Train a personalized model and test it
        self.trainer.test_personalization = True
        personalization_accuracy = self.trainer.test(self.testset)
        self.trainer.test_personalization = False

        if personalization_accuracy == 0:
            # The testing process failed, disconnect from the server
            await self.sio.disconnect()

        logging.info(
            "[%s] Personlization accuracy: %.2f%%", self, 100 * personalization_accuracy
        )

        return personalization_accuracy
