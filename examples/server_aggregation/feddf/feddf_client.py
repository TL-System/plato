"""Client implementation for the FedDF server aggregation example."""

from __future__ import annotations

from feddf_utils import (
    collect_proxy_logits,
    resolve_algorithm_value,
    select_proxy_subset,
)

from plato.clients import simple
from plato.clients.strategies.defaults import DefaultTrainingStrategy


class FedDFTrainingStrategy(DefaultTrainingStrategy):
    """Train locally, then emit teacher logits on a shared proxy set."""

    def __init__(
        self,
        *,
        proxy_size: int | None = None,
        proxy_batch_size: int | None = None,
        proxy_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.proxy_size = proxy_size
        self.proxy_batch_size = proxy_batch_size
        self.proxy_seed = proxy_seed

    async def train(self, context):
        report, _ = await super().train(context)

        datasource = getattr(context, "datasource", None)
        if datasource is None:
            raise RuntimeError("FedDF requires a datasource to resolve proxy samples.")

        trainer = getattr(context, "trainer", None)
        if trainer is None or getattr(trainer, "model", None) is None:
            raise RuntimeError("FedDF requires a trainer with a model for logits.")

        proxy_size = resolve_algorithm_value("proxy_set_size", self.proxy_size, 512)
        proxy_batch_size = resolve_algorithm_value(
            "proxy_batch_size", self.proxy_batch_size, 128
        )
        proxy_seed = resolve_algorithm_value("proxy_seed", self.proxy_seed, 1)

        proxy_dataset, proxy_indices = select_proxy_subset(
            datasource.get_test_set(),
            size=proxy_size,
            seed=proxy_seed,
        )
        logits = collect_proxy_logits(
            trainer.model,
            proxy_dataset,
            batch_size=proxy_batch_size,
            device=getattr(trainer, "device", "cpu"),
        )

        context.state["feddf_proxy_indices"] = proxy_indices
        report.payload_type = "feddf_logits"
        report.proxy_size = len(proxy_indices)

        return report, {"logits": logits}


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks=None,
    proxy_size: int | None = None,
    proxy_batch_size: int | None = None,
    proxy_seed: int | None = None,
):
    """Build a client configured to emit FedDF proxy-set logits."""
    client = simple.Client(
        model=model,
        datasource=datasource,
        algorithm=algorithm,
        trainer=trainer,
        callbacks=callbacks,
        trainer_callbacks=trainer_callbacks,
    )

    client._configure_composable(
        lifecycle_strategy=client.lifecycle_strategy,
        payload_strategy=client.payload_strategy,
        training_strategy=FedDFTrainingStrategy(
            proxy_size=proxy_size,
            proxy_batch_size=proxy_batch_size,
            proxy_seed=proxy_seed,
        ),
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


Client = create_client
