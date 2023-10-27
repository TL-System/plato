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
        """Processing the received payload by assigning modules of local model of
        each client."""

        divergence_scale = data[1]

        # extract the `encoder_modules_name` of the model head
        assert hasattr(Config().algorithm, "encoder_modules_name")

        local_model_modules = self.trainer.model.cpu().state_dict()
        global_modules = data[0]

        global_modules_name = Config().algorithm.global_modules_name
        encoder_modules_name = Config().algorithm.encoder_modules_name

        local_encoder_modules = self.algorithm.get_target_weights(
            model_parameters=local_model_modules, modules_name=encoder_modules_name
        )
        global_encoder_modules = self.algorithm.get_target_weights(
            model_parameters=global_modules, modules_name=encoder_modules_name
        )
        logging.info(
            "[Client #%d] Computing global and local divergence on modules: %s.",
            self.trainer.client_id,
            self.algorithm.extract_modules_name(list(local_encoder_modules.keys())),
        )

        # compute the divergence between encoders of local and global models
        l2_distance = ModelEMA.get_parameters_diff(
            parameter_a=local_encoder_modules,
            parameter_b=global_encoder_modules,
        )

        # EMA update
        divergence_scale = min(l2_distance * divergence_scale, 1)

        # The local model
        local_modules = self.algorithm.get_target_weights(
            model_parameters=local_model_modules, modules_name=global_modules_name
        )
        ema_operator = ModelEMA(beta=divergence_scale)
        ema_parameters = ema_operator.update_parameters_moving_average(
            previous_parameters=local_modules,
            current_parameters=global_modules,
        )
        # update the received payload with the local model as the
        # local model contains all modules
        data[0].update(local_modules)
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
