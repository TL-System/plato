"""Server wrapper for the FedDF server aggregation example."""

from __future__ import annotations

from feddf_algorithm import Algorithm as FedDFAlgorithm
from feddf_server_strategy import FedDFAggregationStrategy

from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using FedDF distillation aggregation."""

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
            aggregation_strategy = FedDFAggregationStrategy()

        selected_algorithm = algorithm or FedDFAlgorithm

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=selected_algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )
