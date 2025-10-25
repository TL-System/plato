"""
An inbound processor for FedEMA to calculate the divergence between received payload
and local saved model weights. And then add on such divergence to the payload.
"""

import logging
from collections.abc import Mapping
from typing import Any

from utils import (
    extract_encoder,
    get_parameters_diff,
    update_parameters_moving_average,
)

from plato.config import Config
from plato.processors import base


class GlobalLocalDivergenceProcessor(base.Processor):
    """
    A processor for clients to compute the divergence between the global
    and the local model.
    """

    def process(self, data: Any) -> Any:
        """Process the received payload by updating the layers using
        the model divergence."""

        if isinstance(data, Mapping):
            if "divergence_rate" not in data:
                raise KeyError("Inbound payload missing 'divergence_rate'.")
            divergence_scale_raw = data["divergence_rate"]
            weights_payload = {
                key: value for key, value in data.items() if key != "divergence_rate"
            }
        elif isinstance(data, (list, tuple)):
            mapping_items = [item for item in data if isinstance(item, Mapping)]
            if not mapping_items:
                raise TypeError(
                    "Expected mapping of model weights in server payload, "
                    f"but received types {[type(item) for item in data]}."
                )
            weights_payload = mapping_items[0]

            non_mapping_items = [item for item in data if item is not weights_payload]
            if not non_mapping_items:
                raise ValueError(
                    "Expected divergence scale accompanying model weights in payload."
                )
            divergence_scale_raw = non_mapping_items[0]
        else:
            raise TypeError(
                "Inbound payload must be a mapping or a sequence of payload components."
            )

        divergence_scale = float(divergence_scale_raw)

        # Extract the `encoder_layer_names` of the model head
        assert hasattr(Config().algorithm, "encoder_layer_names")

        trainer = getattr(self, "trainer", None)
        if trainer is None or trainer.model is None:
            raise RuntimeError(
                "Trainer with a model is required for FedEMA processing."
            )

        local_layers = trainer.model.cpu().state_dict()
        global_layers = dict(weights_payload)

        encoder_layer_names = Config().algorithm.encoder_layer_names

        # Get encoder layers of the local and global models
        local_encoder_layers = extract_encoder(local_layers, encoder_layer_names)
        global_encoder_layers = extract_encoder(global_layers, encoder_layer_names)

        logging.info(
            "[Client #%d] Computing global and local divergence.",
            trainer.client_id,
        )

        # Compute the divergence between encoders of local and global models
        l2_distance = get_parameters_diff(
            parameter_a=local_encoder_layers,
            parameter_b=global_encoder_layers,
        )

        # Perform EMA update
        divergence_scale = float(min(l2_distance * divergence_scale, 1))

        ema_parameters = update_parameters_moving_average(
            previous_parameters=local_layers,
            current_parameters=global_layers,
            beta=divergence_scale,
        )
        # Update the ema parameters
        weights_payload.update(ema_parameters)

        logging.info(
            "[Client #%d] Completed the EMA operation with divergence rate %.3f.",
            trainer.client_id,
            divergence_scale,
        )

        return weights_payload
