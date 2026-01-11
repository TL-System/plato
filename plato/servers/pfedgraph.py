"""
pFedGraph server implementation.
"""

from __future__ import annotations

from typing import Any, Sequence

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.aggregation.pfedgraph import (
    PFedGraphAggregationStrategy,
)


class Server(fedavg.Server):
    """Federated learning server implementing pFedGraph."""

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
            similarity_layers = None
            similarity_metric = "all"
            alpha = 0.8

            if hasattr(Config(), "algorithm"):
                if hasattr(Config().algorithm, "pfedgraph_similarity_metric"):
                    similarity_metric = Config().algorithm.pfedgraph_similarity_metric
                elif hasattr(Config().algorithm, "pfedgraph_similarity"):
                    similarity_metric = Config().algorithm.pfedgraph_similarity

                if hasattr(Config().algorithm, "pfedgraph_similarity_layers"):
                    similarity_layers = Config().algorithm.pfedgraph_similarity_layers

                if hasattr(Config().algorithm, "pfedgraph_alpha"):
                    alpha = Config().algorithm.pfedgraph_alpha

            aggregation_strategy = PFedGraphAggregationStrategy(
                alpha=alpha,
                similarity_metric=similarity_metric,
                similarity_layers=similarity_layers,
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

        self.client_models: dict[int, dict[str, Any]] = {}

    def update_client_model(
        self,
        aggregated_clients_models: Sequence[dict[str, Any]],
        updates: Sequence[Any],
    ) -> None:
        """Update the stored model for each client."""
        for client_model, update in zip(aggregated_clients_models, updates):
            client_id = getattr(update, "client_id", None)
            if client_id is None:
                continue
            self.client_models[client_id] = client_model

    def customize_server_payload(self, payload: Any) -> Any:
        """Send per-client aggregated weights when available."""
        client_id = self.selected_client_id
        if client_id in self.client_models:
            return self.client_models[client_id]
        return payload
