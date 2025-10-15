"""
A federated learning client of Tempo.
"""

from plato.clients import simple
from plato.clients.strategies import DefaultLifecycleStrategy
from plato.config import Config


class TempoLifecycleStrategy(DefaultLifecycleStrategy):
    """Lifecycle strategy that updates local epochs according to server instructions."""

    def process_server_response(self, context, server_response):
        super().process_server_response(context, server_response)
        local_epoch_num = server_response.get("local_epoch_num")
        if local_epoch_num is None:
            return

        Config().trainer = Config().trainer._replace(epochs=local_epoch_num)


class Client(simple.Client):
    """A federated learning client of Tempo."""

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

        payload_strategy = self.payload_strategy
        training_strategy = self.training_strategy
        reporting_strategy = self.reporting_strategy
        communication_strategy = self.communication_strategy

        self._configure_composable(
            lifecycle_strategy=TempoLifecycleStrategy(),
            payload_strategy=payload_strategy,
            training_strategy=training_strategy,
            reporting_strategy=reporting_strategy,
            communication_strategy=communication_strategy,
        )
