"""
pFedGraph server implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from plato.config import Config
from plato.serialization.safetensor import serialize_tree
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

    def _client_model_path(self, client_id: int) -> Path:
        """Return the output path for a saved client-specific pFedGraph model."""

        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        return (
            Path(Config().params["model_path"])
            / f"{model_name}_client_{client_id}.safetensors"
        )

    def save_client_models(self) -> None:
        """Persist the latest pFedGraph client-specific models."""

        if not self.client_models:
            return

        for client_id, client_model in sorted(self.client_models.items()):
            model_path = self._client_model_path(client_id)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with model_path.open("wb") as model_file:
                model_file.write(serialize_tree(client_model))
            logging.info(
                "[%s] Saved pFedGraph client #%d model to %s.",
                self,
                client_id,
                model_path,
            )

    def server_will_close(self) -> None:
        """Save pFedGraph client-specific models before server shutdown."""

        self.save_client_models()
        super().server_will_close()
