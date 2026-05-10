"""Server wrapper for the FedDF server aggregation example."""

from __future__ import annotations

import time

from feddf_algorithm import Algorithm as FedDFAlgorithm
from feddf_server_strategy import FedDFAggregationStrategy
from feddf_utils import stack_proxy_inputs

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
        self.feddf_server_distillation_time = 0.0

    def customize_server_payload(self, payload):
        """Send weights together with the shared proxy inputs for FedDF."""
        proxy_dataset = self.aggregation_strategy._resolve_proxy_dataset(self.context)
        proxy_inputs = self.context.state.get("feddf_proxy_inputs")
        if proxy_inputs is None:
            proxy_inputs = stack_proxy_inputs(proxy_dataset)
        self.context.state["feddf_proxy_inputs"] = proxy_inputs

        return {
            "weights": payload,
            "proxy_inputs": proxy_inputs,
        }

    def clients_processed(self) -> None:
        """Add server distillation time to the simulated round timing."""
        self.feddf_server_distillation_time = float(
            self.context.state.pop("feddf_server_distillation_time", 0.0)
        )

        if self.simulate_wall_time:
            self.wall_time += self.feddf_server_distillation_time
        else:
            self.wall_time = time.time()

    def get_logged_items(self) -> dict:
        """Include FedDF server distillation in logged round metrics."""
        logged = super().get_logged_items()
        logged["processing_time"] += self.feddf_server_distillation_time
        logged["round_time"] += self.feddf_server_distillation_time
        logged["feddf_server_distillation_time"] = self.feddf_server_distillation_time
        return logged
