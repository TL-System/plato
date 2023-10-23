"""
Callback for attaching a pruning mask to the payload if pruning had been conducted.
"""

import os
import pickle
import logging
from typing import OrderedDict


from plato.callbacks.client import ClientCallback
from plato.processors import base
from plato.config import Config


class SendMaskProcessor(base.Processor):
    """
    Implements a processor for attaching a pruning mask to the payload if pruning
    had been conducted
    """

    def process(self, data: OrderedDict):
        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        checkpoint_path = Config().params["checkpoint_path"]

        mask_filename = (
            f"{checkpoint_path}/{model_name}_client{self.client_id}_mask.pth"
        )
        if os.path.exists(mask_filename):
            with open(mask_filename, "rb") as payload_file:
                client_mask = pickle.load(payload_file)
                data = [data, client_mask]
        else:
            data = [data, None]

        if data[1] is not None:
            if self.client_id is None:
                logging.info(
                    "[Server #%d] Pruning mask attached to payload.", self.server_id
                )
            else:
                logging.info(
                    "[Client #%d] Pruning mask attached to payload.", self.client_id
                )
        return data


class HermesCallback(ClientCallback):
    """
    A client callback that dynamically inserts processors into the current list of inbound
    processors.
    """

    def on_outbound_ready(self, client, report, outbound_processor):
        """
        Insert a SendMaskProcessor to the list of outbound processors.
        """
        send_payload_processor = SendMaskProcessor(
            client_id=client.client_id,
            trainer=client.trainer,
            name="SendMaskProcessor",
        )

        outbound_processor.processors.insert(0, send_payload_processor)

        logging.info(
            "[%s] List of outbound processors: %s.",
            client,
            outbound_processor.processors,
        )
