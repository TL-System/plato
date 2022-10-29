"""
A personalized federated learning client.
"""
import logging
import os
import pickle

from plato.clients import simple
from plato.config import Config


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

    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if "personalization_test" in server_response:
            self.do_personalization_test = True

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
