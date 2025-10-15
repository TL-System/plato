"""
A personalized federated learning client that saves its local layers before
sending the shared global model to the server after local training.
"""

from plato.clients import simple
from plato.clients.strategies import FedAvgPersonalizedPayloadStrategy


class Client(simple.Client):
    """
    A personalized federated learning client that saves its local layers before sending the
    shared global model to the server after local training.
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
            payload_strategy=FedAvgPersonalizedPayloadStrategy(),
            training_strategy=self.training_strategy,
            reporting_strategy=self.reporting_strategy,
            communication_strategy=self.communication_strategy,
        )
