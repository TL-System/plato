"""
Customize the processor for FedEMA.
"""
import logging
from typing import Any

import torch

from plato.config import Config
from plato.callbacks.client import ClientCallback
from plato.processors import base

from moving_average import ModelEMA


class GlobalLocalDivergenceProcessor(base.Processor):
    """
    A processor for clients to compute the divergence between the global
    and the local model.
    """

    def __init__(self, algorithm, **kwargs) -> None:
        super().__init__(**kwargs)
        self.algorithm = algorithm

    def process(self, data: Any) -> Any:
        """Processing the received payload by assigning layers of local model of
        each client."""

        divergence_scale = data[1]

        # extract the `encoder_layer_names` of the model head
        assert hasattr(Config().algorithm, "encoder_layer_names")

        local_model_layers = self.trainer.model.cpu().state_dict()
        global_layers = data[0]

        global_layer_names = Config().algorithm.global_layer_names
        encoder_layer_names = Config().algorithm.encoder_layer_names

        local_encoder_layers = self.algorithm.get_target_weights(
            model_parameters=local_model_layers, layer_names=encoder_layer_names
        )
        global_encoder_layers = self.algorithm.get_target_weights(
            model_parameters=global_layers, layer_names=encoder_layer_names
        )
        logging.info(
            "[Client #%d] Computing global and local divergence on layers: %s.",
            self.trainer.client_id,
            self.algorithm.extract_layer_names(list(local_encoder_layers.keys())),
        )

        # compute the divergence between encoders of local and global models
        l2_distance = ModelEMA.get_parameters_diff(
            parameter_a=local_encoder_layers,
            parameter_b=global_encoder_layers,
        )

        # EMA update
        divergence_scale = min(l2_distance * divergence_scale, 1)

        # The local model
        local_layers = self.algorithm.get_target_weights(
            model_parameters=local_model_layers, layer_names=global_layer_names
        )
        ema_operator = ModelEMA(beta=divergence_scale)
        ema_parameters = ema_operator.update_parameters_moving_average(
            previous_parameters=local_layers,
            current_parameters=global_layers,
        )
        # update the ema parameters
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
            algorithm=client.algorithm,
            name="LocalPayloadCompletionProcessor",
        )
        inbound_processor.processors.insert(0, extract_payload_processor)

        logging.info(
            "[%s] List of inbound processors: %s.", client, inbound_processor.processors
        )
