"""
Customize the list of inbound and outbound processors for scaffold clients through callbacks.
"""
import logging
import os
import pickle
from typing import Any, List

from plato.callbacks.client import ClientCallback
from plato.processors import base


class ExtractControlVariatesProcessor(base.Processor):
    """
    A processor for clients to extract the control variates that are attached to the payload
    by the server.
    """

    def __init__(self, client_id, trainer, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id
        self.trainer = trainer

    def process(self, data: List) -> List:
        if len(data) > 1:
            self.trainer.additional_data = data[1]

            logging.info(
                "[Client #%d] Control variates extracted from the payload.",
                self.client_id,
            )
        return data[0]


class SendControlVariateProcessor(base.Processor):
    """
    A processor for clients to attach additional items to the client payload.
    """

    def __init__(self, client_id, trainer, **kwargs) -> None:
        super().__init__(**kwargs)

        self.client_id = client_id
        self.trainer = trainer

    def process(self, data: Any) -> List:
        client_control_variate_filename = self.trainer.client_control_variate_path
        if os.path.exists(client_control_variate_filename):
            with open(client_control_variate_filename, "rb") as payload_file:
                client_control_variate = pickle.load(payload_file)
                data = [data, client_control_variate]

            logging.info(
                "[Client #%d] Control variates were attached to the payload.",
                self.client_id,
            )
        else:
            data = [data, None]

        return data


class ScaffoldCallback(ClientCallback):
    """
    A client callback that dynamically inserts processors into the current list of inbound
    processors.
    """

    def on_inbound_received(self, client, inbound_processor):
        """
        Insert an ExtractPayloadProcessor to the list of inbound processors.
        """
        extract_payload_processor = ExtractControlVariatesProcessor(
            client_id=client.client_id,
            trainer=client.trainer,
            name="ExtractControlVariatesProcessor",
        )
        inbound_processor.processors.insert(0, extract_payload_processor)

        logging.info(
            "[%s] List of inbound processors: %s.", client, inbound_processor.processors
        )

    def on_outbound_ready(self, client, report, outbound_processor):
        """
        Insert a SendControlVariateProcessor to the list of outbound processors.
        """
        send_payload_processor = SendControlVariateProcessor(
            client_id=client.client_id,
            trainer=client.trainer,
            name="SendControlVariateProcessor",
        )

        outbound_processor.processors.insert(0, send_payload_processor)

        logging.info(
            "[%s] List of outbound processors: %s.",
            client,
            outbound_processor.processors,
        )
