"""
Strategies for FedAvg personalized clients.

These strategies enable the composable client architecture to run the
FedAvg-personalized workflow while keeping local layer persistence in sync
with the legacy implementation.
"""

from __future__ import annotations

from collections import OrderedDict

from plato.clients.strategies.base import ClientContext
from plato.clients.strategies.defaults import DefaultPayloadStrategy
from plato.config import Config


class FedAvgPersonalizedPayloadStrategy(DefaultPayloadStrategy):
    """Payload strategy that persists local layers before outbound processing."""

    def outbound_ready(
        self,
        context: ClientContext,
        report,
        outbound_payload,
    ) -> None:
        owner = context.owner
        super().outbound_ready(context, report, outbound_payload)

        if owner is None:
            return

        weights = context.algorithm.extract_weights()

        if hasattr(Config().algorithm, "local_layer_names"):
            local_layers = OrderedDict(
                [
                    (name, param)
                    for name, param in weights.items()
                    if any(
                        param_name in name.strip().split(".")
                        for param_name in Config().algorithm.local_layer_names
                    )
                ]
            )

            model_path = Config().params["model_path"]
            model_name = Config().trainer.model_name
            filename = f"{model_path}/{model_name}_{context.client_id}_local_layers.pth"

            context.algorithm.save_local_layers(local_layers, filename)
