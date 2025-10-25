"""
A federated learning server using Port.

Reference:

"How Asynchronous can Federated Learning Be?"

"""

from plato.servers import fedavg
from plato.servers.strategies import PortAggregationStrategy


class Server(fedavg.Server):
    """A federated learning server using the Port aggregation strategy."""

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
            aggregation_strategy=PortAggregationStrategy(),
        )

    def weights_aggregated(self, updates):
        """
        Method called at the end of aggregating received weights.
        """
        # Save the current model for later retrieval when cosine similarity needs to be computed
        filename = f"model_{self.current_round}.pth"
        trainer = self.require_trainer()
        trainer.save_model(filename)
