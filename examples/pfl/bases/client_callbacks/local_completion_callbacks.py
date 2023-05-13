"""
Customize client callbacks for self-supervised learning to complete the received 
payload with parameters of local model. The local model is the locally 
trained global model in the previous round.
For instance, when payload contains the body of one model, the head 
of the loaded local model trained in the previous round will be used to 
complete the payload.
"""
import os

import logging
from typing import Any

from plato.config import Config
from plato.processors import base
from plato.utils.filename_formatter import NameFormatter

from bases.client_callbacks import base_callbacks


class PayloadCompletionProcessor(base.Processor):
    """A processor relying on the hyper-parameter `completion_modules_name`
    to complete parameters of payload with the loaded personalized model."""

    def __init__(self, current_round, algorithm, **kwargs) -> None:
        super().__init__(**kwargs)
        self.current_round = current_round
        self.algorithm = algorithm

    def process(self, data: Any) -> Any:
        """Processing the received payload by assigning modules of personalized model
        if provided."""
        model_name = Config().trainer.model_name
        checkpoint_dir_path = self.trainer.get_checkpoint_dir_path()

        # local the locally saved model from the previous round
        desired_round = self.current_round - 1
        filename = NameFormatter.get_format_name(
            model_name=model_name,
            client_id=self.trainer.client_id,
            round_n=0,
            epoch_n=None,
            run_id=None,
            prefix=None,
            ext="pth",
        )

        checkpoint_file_path = os.path.join(checkpoint_dir_path, filename)

        # if the desired of trained model is not saved by this client
        # this client is never selected to perform the training
        if not os.path.exists(checkpoint_file_path):
            self.trainer.model.apply(self.trainer.reset_weight)
            self.trainer.save_model(filename=filename, location=checkpoint_dir_path)
        else:
            loaded_model = self.trainer.rollback_model(
                rollback_round=desired_round,
                location=checkpoint_dir_path,
                model_name=model_name,
                modelfile_prefix="",
            )
            self.trainer.model.load_state_dict(loaded_model, strict=True)

        # extract the `completion_modules_name` of the model head
        assert hasattr(Config().algorithm, "completion_modules_name")

        completion_modules_name = Config().algorithm.completion_modules_name
        model_modules = self.algorithm.extract_weights(
            model=self.trainer.model, modules_name=completion_modules_name
        )
        logging.info(
            "[Client #%d] Extracted modules: %s from its loaded local model.",
            self.trainer.client_id,
            self.algorithm.extract_modules_name(list(model_modules.keys())),
        )

        data.update(model_modules)

        logging.info(
            "[Client #%d] Completed the payload with extracted modules.",
            self.trainer.client_id,
        )

        return data


class ClientModelLocalCompletionCallback(base_callbacks.ClientPayloadCallback):
    """
    A client callback for FedBABU approach to process the received payload.
    """

    def on_inbound_received(self, client, inbound_processor):
        """Completing the recevied payload with the personalized model."""
        super().on_inbound_received(client, inbound_processor)

        inbound_processor.processors.append(
            PayloadCompletionProcessor(
                current_round=client.current_round,
                trainer=client.trainer,
                algorithm=client.algorithm,
            )
        )
