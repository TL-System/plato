"""Client implementation for the FedDF server aggregation example."""

from __future__ import annotations

import time

from feddf_utils import (
    collect_proxy_logits,
    resolve_algorithm_value,
)
from torch.utils.data import TensorDataset

from plato.clients import simple
from plato.clients.strategies.defaults import DefaultTrainingStrategy


class FedDFTrainingStrategy(DefaultTrainingStrategy):
    """Train locally, then emit teacher logits on a shared proxy set."""

    def __init__(
        self,
        *,
        proxy_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.proxy_batch_size = proxy_batch_size

    def load_payload(self, context, server_payload) -> None:
        """Load model weights and cache the shared proxy inputs from the server."""
        if not isinstance(server_payload, dict):
            raise TypeError("FedDF expects a dictionary payload from the server.")

        if "weights" not in server_payload or "proxy_inputs" not in server_payload:
            raise KeyError(
                "FedDF server payload must include 'weights' and 'proxy_inputs'."
            )

        context.state["feddf_proxy_inputs"] = server_payload["proxy_inputs"]
        super().load_payload(context, server_payload["weights"])

    async def train(self, context):
        report, _ = await super().train(context)

        trainer = getattr(context, "trainer", None)
        if trainer is None or getattr(trainer, "model", None) is None:
            raise RuntimeError("FedDF requires a trainer with a model for logits.")

        proxy_inputs = context.state.get("feddf_proxy_inputs")
        if proxy_inputs is None:
            raise RuntimeError("FedDF requires shared proxy inputs from the server.")

        proxy_batch_size = resolve_algorithm_value(
            "proxy_batch_size", self.proxy_batch_size, 128
        )
        tic = time.perf_counter()
        logits = collect_proxy_logits(
            trainer.model,
            TensorDataset(proxy_inputs),
            batch_size=proxy_batch_size,
            device=getattr(trainer, "device", "cpu"),
        )
        proxy_logits_time = time.perf_counter() - tic

        report.training_time = getattr(report, "training_time", 0.0) + proxy_logits_time
        report.feddf_proxy_logits_time = proxy_logits_time
        context.state["feddf_proxy_logits_time"] = proxy_logits_time

        report.payload_type = "feddf_logits"
        report.proxy_size = len(proxy_inputs)

        return report, {"logits": logits}


def create_client(
    *,
    model=None,
    datasource=None,
    algorithm=None,
    trainer=None,
    callbacks=None,
    trainer_callbacks=None,
    proxy_batch_size: int | None = None,
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
            proxy_batch_size=proxy_batch_size,
        ),
        reporting_strategy=client.reporting_strategy,
        communication_strategy=client.communication_strategy,
    )

    return client


Client = create_client
