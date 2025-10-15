"""
A federated learning client at the edge server in a cross-silo training workload.
"""

from plato.clients import simple
from plato.clients.strategies import EdgeLifecycleStrategy, EdgeTrainingStrategy


class Client(simple.Client):
    """A federated learning client at the edge server in a cross-silo training workload."""

    def __init__(
        self,
        server,
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
        self.server = server

        self._configure_composable(
            lifecycle_strategy=EdgeLifecycleStrategy(),
            payload_strategy=self.payload_strategy,
            training_strategy=EdgeTrainingStrategy(),
            reporting_strategy=self.reporting_strategy,
            communication_strategy=self.communication_strategy,
        )
