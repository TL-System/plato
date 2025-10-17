"""
A personalized federated learning client.
"""

import logging
import os

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultTrainingStrategy
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
            training_strategy=FlMamlTrainingStrategy(),
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )


class FlMamlTrainingStrategy(DefaultTrainingStrategy):
    """Training strategy that supports personalization tests for FL MAML."""

    async def train(self, context: ClientContext):
        owner = context.owner
        if owner is not None and getattr(owner, "do_personalization_test", False):
            owner.do_personalization_test = False
            report = await self._test_personalized_model(context)
            payload = "personalization_accuracy"
            return report, payload

        report, payload = await super().train(context)
        if Config().is_edge_server():
            logging.info(
                "[Server #%d] Model aggregated on edge server (%s).",
                os.getpid(),
                owner,
            )
        else:
            logging.info("[%s] Model trained.", owner)

        return report, payload

    async def _test_personalized_model(self, context: ClientContext):
        logging.info("[%s] Started training a personalized model.", context.owner)

        trainer = context.trainer
        if trainer is None:
            raise RuntimeError("Trainer is required for personalization testing.")

        trainer.test_personalization = True
        personalization_accuracy = trainer.test(context.testset)
        trainer.test_personalization = False

        if personalization_accuracy == 0 and context.sio is not None:
            await context.sio.disconnect()

        logging.info(
            "[%s] Personlization accuracy: %.2f%%",
            context.owner,
            100 * personalization_accuracy,
        )

        return personalization_accuracy
