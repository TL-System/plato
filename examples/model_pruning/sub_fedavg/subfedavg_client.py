"""
A federated learning client using pruning in Sub-FedAvg.
"""

import logging

from plato.clients import simple
from plato.clients.strategies.defaults import DefaultTrainingStrategy


class SubFedAvgTrainingStrategy(DefaultTrainingStrategy):
    """Training strategy that logs Sub-FedAvg completion."""

    async def train(self, context):
        report, weights = await super().train(context)
        logging.info(
            "[Client #%d] Trained with Sub-FedAvg algorithm.", context.client_id
        )
        return report, weights


class Client(simple.Client):
    """
    A federated learning client prunes its update before sending out.
    """

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            trainer_callbacks=trainer_callbacks,
        )

        self._configure_composable(
            lifecycle_strategy=self.lifecycle_strategy,
            payload_strategy=self.payload_strategy,
            training_strategy=SubFedAvgTrainingStrategy(),
            reporting_strategy=self.reporting_strategy,
            communication_strategy=self.communication_strategy,
        )
