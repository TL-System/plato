"""
Server orchestration for MOON using the new aggregation strategy API.
"""

from __future__ import annotations

from plato.servers import fedavg

from .moon_server_strategy import MoonAggregationStrategy


class Server(fedavg.Server):
    """Federated server using the MOON aggregation strategy by default."""

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
        aggregation_strategy = aggregation_strategy or MoonAggregationStrategy()

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )
