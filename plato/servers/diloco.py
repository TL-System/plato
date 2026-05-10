"""FedAvg-compatible server using DiLoCo aggregation."""

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.aggregation import DiLoCoAggregationStrategy


class Server(fedavg.Server):
    """Federated learning server with server-side DiLoCo outer aggregation."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        aggregation_strategy=None,
        client_selection_strategy=None,
    ):
        if aggregation_strategy is None:
            aggregation_strategy = DiLoCoAggregationStrategy(
                **self._aggregation_config()
            )

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )

    @staticmethod
    def _aggregation_config() -> dict:
        """Read optional DiLoCo aggregation settings from [server.diloco]."""
        config = getattr(Config().server, "diloco", None)
        if config is None:
            return {}

        keys = (
            "outer_optimizer",
            "outer_learning_rate",
            "outer_momentum",
            "aggregation_weighting",
            "apply_outer_optimizer_to",
        )
        return {key: getattr(config, key) for key in keys if hasattr(config, key)}
