"""
An inbound processor for FedEMA to calculate the divergence between received payload
and local saved model weights. And then add on such divergence to the payload.
"""

import logging
from typing import Any

import utils

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

        divergence_scale = data[1]

        # Extract the `encoder_layer_names` of the model head
        assert hasattr(Config().algorithm, "encoder_layer_names")

        local_layers = self.trainer.model.cpu().state_dict()
        global_layers = data[0]

        encoder_layer_names = Config().algorithm.encoder_layer_names

        # Get encoder layers of the local and global models
        local_encoder_layers = utils.extract_encoder(local_layers, encoder_layer_names)
        global_encoder_layers = utils.extract_encoder(
            global_layers, encoder_layer_names
        )

        logging.info(
            "[Client #%d] Computing global and local divergence.",
            self.trainer.client_id,
        )

        # Compute the divergence between encoders of local and global models
        l2_distance = utils.get_parameters_diff(
            parameter_a=local_encoder_layers,
            parameter_b=global_encoder_layers,
        )

        # Perform EMA update
        divergence_scale = min(l2_distance * divergence_scale, 1)

        ema_parameters = utils.update_parameters_moving_average(
            previous_parameters=local_layers,
            current_parameters=global_layers,
            beta=divergence_scale,
        )
        # Update the ema parameters
        data[0].update(ema_parameters)

        logging.info(
            "[Client #%d] Completed the EMA operation with divergence rate %.3f.",
            self.trainer.client_id,
            divergence_scale,
        )

        return data[0]
