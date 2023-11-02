"""
Customize the processor for FedEMA.
"""
import logging
from typing import Any

import utils
from moving_average import ModelEMA

from plato.config import Config
from plato.callbacks.client import ClientCallback
from plato.processors import base


class GlobalLocalDivergenceProcessor(base.Processor):
    """
    A processor for clients to compute the divergence between the global
    and the local model.
    """

    def process(self, data: Any) -> Any:
        """Processing the received payload by assigning layers of local model of
        each client."""

        divergence_scale = data[1]

        # Extract the `encoder_layer_names` of the model head
        assert hasattr(Config().algorithm, "encoder_layer_names")

        local_model_layers = self.trainer.model.cpu().state_dict()
        global_layers = data[0]

        encoder_layer_names = Config().algorithm.encoder_layer_names

        # Get encoder layers of the local and global models
        local_encoder_layers = utils.extract_encoder(
            local_model_layers, encoder_layer_names
        )
        global_encoder_layers = utils.extract_encoder(
            global_layers, encoder_layer_names
        )

        logging.info(
            "[Client #%d] Computing global and local divergence.",
            self.trainer.client_id,
        )

        # Compute the divergence between encoders of local and global models
        l2_distance = ModelEMA.get_parameters_diff(
            parameter_a=local_encoder_layers,
            parameter_b=global_encoder_layers,
        )

        # Perform EMA update
        divergence_scale = min(l2_distance * divergence_scale, 1)

        ema_operator = ModelEMA(beta=divergence_scale)
        ema_parameters = ema_operator.update_parameters_moving_average(
            previous_parameters=local_model_layers,
            current_parameters=global_layers,
        )
        # Update the ema parameters
        data[0].update(ema_parameters)

        logging.info(
            "[Client #%d] Completed the EMA operation with divergence rate %.3f.",
            self.trainer.client_id,
            divergence_scale,
        )

        return data[0]


class FedEMACallback(ClientCallback):
    """
    A client callback that dynamically compute the divergence between the received model
    and the local model.
    """

    def on_inbound_received(self, client, inbound_processor):
        """
        Insert an GlobalLocalDivergenceProcessor to the list of inbound processors.
        """
        extract_payload_processor = GlobalLocalDivergenceProcessor(
            trainer=client.trainer,
            name="GlobalLocalDivergenceProcessor",
        )
        inbound_processor.processors.insert(0, extract_payload_processor)
