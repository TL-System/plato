"""FedDF aggregation strategy using server-side proxy-set distillation."""

from __future__ import annotations

from typing import Mapping

from feddf_utils import resolve_algorithm_value, select_proxy_subset
from torch.utils.data import Dataset

from plato.datasources import registry as datasources_registry
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedDFAggregationStrategy(AggregationStrategy):
    """Aggregate client logits and distill them into the global student."""

    def __init__(
        self,
        *,
        proxy_set_size: int | None = None,
        proxy_seed: int | None = None,
        temperature: float | None = None,
        distillation_epochs: int | None = None,
        distillation_batch_size: int | None = None,
        distillation_learning_rate: float | None = None,
    ) -> None:
        super().__init__()
        self.proxy_set_size = proxy_set_size
        self.proxy_seed = proxy_seed
        self.temperature = temperature
        self.distillation_epochs = distillation_epochs
        self.distillation_batch_size = distillation_batch_size
        self.distillation_learning_rate = distillation_learning_rate

    async def aggregate_deltas(self, updates, deltas_received, context: ServerContext):
        """FedDF does not aggregate parameter deltas."""
        raise NotImplementedError("FedDF uses aggregate_weights with logits payloads.")

    @staticmethod
    def _proxy_source_dataset(datasource):
        """Prefer an unlabeled proxy split, falling back to the test split."""
        if hasattr(datasource, "get_unlabeled_set"):
            unlabeled_set = datasource.get_unlabeled_set()
            if unlabeled_set is not None:
                return unlabeled_set

        test_set = datasource.get_test_set()
        if test_set is None:
            raise RuntimeError(
                "FedDF requires either an unlabeled proxy split or a test split."
            )
        return test_set

    def _resolve_proxy_dataset(self, context: ServerContext) -> Dataset:
        """Construct or reuse the deterministic proxy subset."""
        cached = context.state.get("feddf_proxy_dataset")
        if cached is not None:
            return cached

        server = getattr(context, "server", None)
        if server is None:
            raise RuntimeError("FedDF requires the server in strategy context.")

        datasource = getattr(server, "datasource", None)
        if datasource is None:
            custom_datasource = getattr(server, "custom_datasource", None)
            if custom_datasource is not None:
                datasource = custom_datasource()
            else:
                datasource = datasources_registry.get(client_id=0)
            server.datasource = datasource

        proxy_set_size = resolve_algorithm_value(
            "proxy_set_size", self.proxy_set_size, 512
        )
        proxy_seed = resolve_algorithm_value("proxy_seed", self.proxy_seed, 1)
        proxy_dataset, proxy_indices = select_proxy_subset(
            self._proxy_source_dataset(datasource),
            size=proxy_set_size,
            seed=proxy_seed,
        )

        context.state["feddf_proxy_dataset"] = proxy_dataset
        context.state["feddf_proxy_indices"] = proxy_indices
        return proxy_dataset

    async def aggregate_weights(
        self,
        updates,
        baseline_weights: Mapping,
        weights_received,
        context: ServerContext,
    ):
        """Distill the global student from client logits on the proxy set."""
        algorithm = getattr(context, "algorithm", None)
        if algorithm is None:
            raise RuntimeError("FedDF requires an algorithm instance in context.")

        proxy_dataset = self._resolve_proxy_dataset(context)
        teacher_logits = algorithm.aggregate_teacher_logits(updates, weights_received)

        temperature = resolve_algorithm_value("temperature", self.temperature, 2.0)
        distillation_epochs = resolve_algorithm_value(
            "distillation_epochs", self.distillation_epochs, 5
        )
        distillation_batch_size = resolve_algorithm_value(
            "distillation_batch_size", self.distillation_batch_size, 64
        )
        distillation_learning_rate = resolve_algorithm_value(
            "learning_rate", self.distillation_learning_rate, 0.001
        )

        return algorithm.distill_weights(
            baseline_weights,
            teacher_logits,
            proxy_dataset,
            temperature=temperature,
            distillation_epochs=distillation_epochs,
            distillation_batch_size=distillation_batch_size,
            distillation_learning_rate=distillation_learning_rate,
        )
